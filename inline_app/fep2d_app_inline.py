"""
fep2d_app_inline.py  -  2D PXRD INLINE analysis app (Streamlit), for use at the beamline
=========================================================================================
Run:    streamlit run fep2d_app_inline.py
Also:   python inline_worker.py ...   (started from the sidebar button, or by hand in its own terminal)
Needs:  inline_worker.py, live_pattern.py, fep2d_v3.py and fep2d_pipeline.py in the same folder (plus pages/Live_viewer.py), the frozen ML_version1_2 classifier folder,
        plus numpy, scipy, h5py, pyFAI, pybaselines, matplotlib, pandas, plotly, streamlit (1.37 or newer)

How it fits together
    The engineer's script writes  I11-1-<collection number>_subtracted.npy  into the WATCHED folder.
    inline_worker.py (a separate process) finds each new frame once it is complete, classifies it, integrates
    it and logs it.  This app never watches the folder itself: it only reads the small files the worker writes,
    so it stays quick however many frames arrive.

Sidebar   run name, run events (conditions changed / run finished), open the live viewer, output folder, watched folder, calibration status, extra 2-theta mask, integration and
          baseline settings, classifier cut-offs, worker start / stop and live status
Tab 0     Start-up           calibration, merged reference maker, how-many-reference-frames test
Tab 1     Watcher            live counts, score timeline, newest frames, worker log
Tab 2     Visualiser + sorting   ML results; review the unsure frames by eye (your label always wins)
Tab 3     Integrate + baseline   MOR baseline of the stored 1D patterns, merged 1D pattern for the run
Tab 4     Pattern stacker    unchanged from the offline app
Pop-out   pages/Live_viewer.py   live cumulative 1D pattern in its own window, with reference overlays (sidebar button)

Output layout, under <output folder>/<run name>/:
    00_reference/   merged_reference.npz, reference_size_test.csv
    01b_sorting/    ml_scores.csv (worker), sorting_decisions.csv (this app), hit_list.txt
    02_integrated/  one .xy per frame (written by the worker)
    03_baselined/   one .xy per frame, MOR baselined
    04_merged/      merged patterns; one cumulative final pattern per segment (<run>_seg<k>_...)
    run_events.csv  your markers (conditions changed / new reference / run finished) with notes
    worker_heartbeat.json, worker_manifest.json, worker.log, session_settings.json
"""
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fep2d_pipeline as fp  # noqa: E402
import fep2d_v3 as fm  # noqa: E402
import live_pattern as lp  # noqa: E402

APP_VERSION = "fep2d_app_inline v2 / inline_worker v3 / live_pattern v1 / fep2d_pipeline v4 / fep2d_v3 (V2 maths unchanged)"
MASK_BLUE = "#0A84FF"   # masked pixels in the sorting tab frame view
DEFAULT_MASK_TEXT = fp.fmt_ranges(fp.DEFAULT_EXTRA_MASK)        # "5.30-6.10"
SETTINGS_FILE = Path(__file__).resolve().parent / "inline_last_session.json"
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="2D PXRD inline", page_icon="🧪", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 20px; border-radius: 6px; }
    .status-ok  { color: #1a7a4a; font-weight: 600; }
    .status-warn{ color: #e67e22; font-weight: 600; }
    .status-bad { color: #c0392b; font-weight: 700; }
</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clean_path(s) -> str:
    return str(s or "").strip().strip('"').strip("'")


def pick_path(mode="folder", title="Select", filetypes=None, initialdir=None):
    """Native file/folder dialog via tkinter (same as the 1D app). Returns a path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        kw = {"initialdir": initialdir} if initialdir and os.path.isdir(initialdir) else {}
        if mode == "folder":
            path = filedialog.askdirectory(title=title, **kw)
        else:
            path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All files", "*.*")], **kw)
        root.destroy()
        return path or None
    except Exception as e:                      # no display, no tkinter, etc.
        st.warning(f"The browse dialog could not open ({e}). Type the path instead.")
        return None


def path_input(label, key, mode="folder", placeholder="", help=None, filetypes=None):
    """A text box plus a Browse button. The text box is the source of truth, so a path can always
    be typed or pasted. Returns the cleaned path."""
    pend = st.session_state.pop(f"_pending_{key}", None)
    if pend:
        st.session_state[key] = pend
    st.session_state.setdefault(key, "")
    c1, c2 = st.columns([6, 1])
    c1.text_input(label, key=key, placeholder=placeholder, help=help)
    c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if c2.button("📂", key=f"_btn_{key}", help="Browse", use_container_width=True):
        got = pick_path(mode, f"Select: {label}", filetypes, clean_path(st.session_state.get(key)) or None)
        if got:
            st.session_state[f"_pending_{key}"] = got
            st.rerun()
    return clean_path(st.session_state.get(key))


def _df(df, **kw):
    try:
        return st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        return st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def _pyplot(fig):
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def exts_from_choice(choice: str):
    return {".nxs": (".nxs",), ".hdf": (".hdf", ".h5", ".hdf5"),
            "both (.nxs + .hdf)": (".nxs", ".hdf", ".h5", ".hdf5")}[choice]


def block_max(a, b=4):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).max(axis=(1, 3))


def show_image(ax, a, vmin, vmax, title, cmap="gray", mask=None):
    img = np.where(mask, np.nan, a) if mask is not None else a
    ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


@st.cache_resource(show_spinner=False)
def _load_ref(path: str, mtime: float):
    return fm.load_reference(path)


def get_ref(path: str):
    path = clean_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference file not found: {path}")
    return _load_ref(path, os.path.getmtime(path))


def _geom_key(ai, mask_total):
    """Cache key made from the loaded calibration and mask themselves, never from file paths
    (a blank or stale path box must not break the ring map)."""
    g = tuple(round(float(v), 12) for v in (ai.dist, ai.poni1, ai.poni2, ai.rot1, ai.rot2, ai.rot3,
                                            ai.wavelength or 0.0, ai.pixel1, ai.pixel2))
    m = np.ascontiguousarray(mask_total[::37, ::41])
    return g, tuple(mask_total.shape), int(mask_total.sum()), hash(m.tobytes())


@st.cache_resource(show_spinner="Building the ring map (once per geometry and mask)...")
def _ring_map_cached(geom_key, ranges_key, _ai, _mask_total):
    return fm.build_ring_map(_ai, _mask_total.shape, _mask_total)


def calib():
    """(ai, detector mask) from the Calibration tab, or raises with a clear message."""
    if not st.session_state.calib_loaded:
        raise RuntimeError("Load and verify the calibration first (tab 0).")
    return st.session_state.ai, st.session_state.mask


def extra_ranges():
    """The extra 2-theta mask ranges from the sidebar, or [] if switched off."""
    if not st.session_state.use_extra_mask:
        return []
    return fp.parse_ranges(st.session_state.extra_mask_text)


def stage_dir(name: str):
    """<output folder>/<run name>/<name>, or None if no output folder is set."""
    out = clean_path(st.session_state.output_dir)
    if not out:
        return None
    return Path(out) / (clean_path(st.session_state.run_name) or "run") / name


def need_output(name: str, typed: str = "") -> str:
    """Folder to write to: the typed path, else the stage folder in the sidebar's output folder."""
    if typed:
        return typed
    d = stage_dir(name)
    if d is None:
        raise ValueError("Set the output folder in the sidebar (or type a folder here).")
    return str(d)



# ---------------------------------------------------------------------------
# inline helpers: worker files, scores, decisions, classifier
# ---------------------------------------------------------------------------
def live_fragment(seconds):
    """Decorator: re-run only this block every few seconds (Streamlit 1.37+). Falls back to a plain block."""
    f = getattr(st, "fragment", None)
    if f is None:
        return lambda fn: fn
    try:
        return f(run_every=seconds)
    except TypeError:
        return f


def run_dir():
    return stage_dir("")


def frame_exts():
    return [e.strip().lower().lstrip(".") for e in str(st.session_state.get("frame_exts", "npy,hdf,nxs")).split(",") if e.strip()] or ["npy"]


def frame_pat():
    """File names the worker picks up: <prefix><collection number>[<suffix>].<ext>  (suffix optional, not case sensitive)."""
    pre = re.escape(st.session_state.prefix) if st.session_state.prefix else r".*?"
    suf = f"(?:{re.escape(st.session_state.suffix)})?" if st.session_state.suffix else ""
    return re.compile(rf"^{pre}(\d+){suf}\.({'|'.join(map(re.escape, frame_exts()))})$", re.IGNORECASE)


def hb_path():
    d = run_dir()
    return None if d is None else d / "worker_heartbeat.json"


def read_heartbeat():
    try:
        p = hb_path()
        if p is not None and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def ml_scores_path():
    d = stage_dir("01b_sorting")
    return None if d is None else d / "ml_scores.csv"


SCORE_COLS = ["file", "collection", "score", "ml_label", "lower", "upper", "guard", "error", "sec", "saved"]


def read_scores_df():
    """ml_scores.csv written by the worker (latest row per frame, in collection order). If the file is being
    written at that instant, the last good copy is returned."""
    try:
        p = ml_scores_path()
        if p is not None and p.is_file():
            df = pd.read_csv(p).drop_duplicates("file", keep="last")
            for c in SCORE_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["guard"] = df["guard"].fillna("").astype(str)
            df = df.sort_values("collection").reset_index(drop=True)
            st.session_state["_last_scores"] = df
            return df
    except Exception:
        if st.session_state.get("_last_scores") is not None:
            return st.session_state["_last_scores"]
    return pd.DataFrame(columns=SCORE_COLS)


SORT_LABELS = {"hit": "Hit", "non_hit": "Non-hit", "unsure": "Unsure", "bad": "Bad frame"}


def sort_csv_path():
    d = stage_dir("01b_sorting")
    return None if d is None else d / "sorting_decisions.csv"


def read_decisions_csv(path):
    try:
        if path is not None and Path(path).is_file():
            df_ = pd.read_csv(path)
            return {str(r.file): str(r.label) for r in df_.itertuples() if str(r.label) in SORT_LABELS}
    except Exception:
        pass
    return {}


def save_decisions(dec, folder=""):
    """Write the decisions csv after every click so nothing is lost. False if no output folder."""
    p = sort_csv_path()
    if p is None:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(file=k, label=v, folder=folder, saved=datetime.datetime.now().isoformat(timespec="seconds"))
            for k, v in sorted(dec.items())]
    tmp = p.with_suffix(".tmp")
    pd.DataFrame(rows, columns=["file", "label", "folder", "saved"]).to_csv(tmp, index=False)
    os.replace(tmp, p)
    return True


def effective_labels(df=None):
    """{file: label}: your decision if you made one, otherwise the ML label. Read fresh from disk."""
    df = read_scores_df() if df is None else df
    dec = read_decisions_csv(sort_csv_path())
    return {str(r.file): dec.get(str(r.file), str(r.ml_label)) for r in df.itertuples()}, dec, df


ML_FROZEN_VERSION = "1.2"


def _ml_default_dir():
    """Where the frozen Version 1.2 classifier is looked for when the box is empty: the ML_V1_DIR environment variable, a
    ML_version1_2 folder next to this app, then the AP40 beamline pack on the thesis drive. The old Version 1 folder is NOT used."""
    for c in (os.environ.get("ML_V1_DIR"), str(APP_DIR / "ML_version1_2"),
              r"E:/2D_sub_appV4_trial/ML+AppDevelopment/beamline_pack_AP40/ML_version1_2"):
        if c and os.path.isfile(os.path.join(c, "classify_v1.py")):
            return c
    return r"E:/2D_sub_appV4_trial/ML+AppDevelopment/beamline_pack_AP40/ML_version1_2"


def model_info(folder):
    """What is really in a classifier folder: version, training size, cut-offs, and anything missing."""
    folder = clean_path(folder) or _ml_default_dir()
    info = dict(folder=folder, ok=False, version=None, rows=None, lower=None, upper=None, missing=[], note="")
    need = ["classify_v1.py", "features_v1.py", "record.json", "config.json", "frame_guard.py", "spike_guard.py", "run_start_basic.py"]
    info["missing"] = [f for f in need if not os.path.isfile(os.path.join(folder, f))]
    if not os.path.isdir(os.path.join(folder, "model")):
        info["missing"].append("model/")
    try:
        r = json.load(open(os.path.join(folder, "record.json"), encoding="utf-8"))
        c = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        info.update(version=str(r.get("version")), rows=r.get("n_frames"), lower=c.get("lower"), upper=c.get("upper"))
    except Exception:
        pass
    info["ok"] = not info["missing"] and info["version"] is not None
    return info


@st.cache_resource(show_spinner="Loading the classifier (used for the run-start check only)...")
def load_ml_v1(folder):
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from classify_v1 import ClassifierV1
    return ClassifierV1(folder=folder)


def _ml_mods(folder):
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import run_start_basic as _rs
    import frame_guard as _fg
    return _rs, _fg


H5_PATHS = ("entry1/pixium_hdf/data", "entry/data/data", "entry1/data/data", "entry1/Pixium10:detector/data",
            "entry1/detector/data", "entry/detector/data")


def read_frame_file(path):
    """One 2D float32 frame from .npy, or from .hdf / .h5 / .nxs (first frame of a 3D dataset)."""
    path = str(path)
    if path.lower().endswith(".npy"):
        a = np.load(path, allow_pickle=False)
    else:
        import h5py
        with h5py.File(path, "r") as f:
            p = next((q for q in H5_PATHS if q in f), None)
            if p is None:
                found = []
                f.visititems(lambda n, o: found.append(n) if hasattr(o, "ndim") and o.ndim >= 2 else None)
                p = max(found, key=lambda n: int(np.prod(f[n].shape[-2:])))
            a = f[p][()]
    if a.ndim == 3:
        a = a[0]
    return np.ascontiguousarray(a, dtype=np.float32)


@st.cache_data(show_spinner=False, max_entries=3)
def load_npy_cached(path, mtime):
    return read_frame_file(path)


def xy_for(name):
    """The worker's stored 1D pattern for a frame, or None."""
    d = stage_dir("02_integrated")
    if d is None:
        return None
    p = d / f"{Path(name).stem}.xy"
    return p if p.is_file() else None


def worker_command(rd):
    ss = st.session_state
    cmd = [sys.executable, str(APP_DIR / "inline_worker.py"),
           "--watch", clean_path(ss.watch_dir), "--out", str(rd),
           "--poni", clean_path(ss.poni_path), "--mask", clean_path(ss.mask_path),
           "--ml-dir", clean_path(ss.ml_v1_dir) or _ml_default_dir(), "--pipeline-dir", str(APP_DIR),
           "--prefix", ss.prefix, "--suffix", ss.suffix, "--ext", ",".join(frame_exts()), "--poll", str(float(ss.poll_s)),
           "--extra-mask", (clean_path(ss.extra_mask_text) if ss.use_extra_mask else ""),
           "--npt", str(int(ss.npt)), "--rmin", str(float(ss.rmin)), "--rmax", str(float(ss.rmax))]
    if ss.ml_override:
        cmd += ["--lower", str(float(ss.ml_lo)), "--upper", str(float(ss.ml_hi))]
    return cmd




# ---------------------------------------------------------------------------
# run events: conditions changed / run finished (segments of one run)
# ---------------------------------------------------------------------------
PER_RUN_STATE = ("sort_dec", "sort_snap", "int_result", "mg_result", "rs_result", "_last_scores", "ev_prompt")


def current_segments():
    sc = read_scores_df()
    rd = run_dir()
    ev = lp.read_events(rd) if rd is not None else lp.read_events("")
    return sc, ev, lp.segments(sc, ev)


def mark_event():
    """on_click callback of the 'Mark it' button."""
    ss = st.session_state
    rd = run_dir()
    ss["ev_msg"] = None
    if rd is None or not clean_path(ss.run_name):
        ss["ev_msg"] = ("error", "Set the output folder and run name first.")
        return
    sc = read_scores_df()
    if sc.empty:
        ss["ev_msg"] = ("error", "No frames have been scored yet, so there is nothing to mark.")
        return
    coll = int(ss.get("ev_coll", 0) or 0) or int(sc["collection"].max())
    kind = ss.ev_kind
    note = str(ss.get("ev_note", "")).strip()
    lp.append_event(rd, coll, kind, note, clean_path(ss.run_name))
    try:
        _eff, _dec, _df0 = effective_labels(sc)
        segs = lp.segments(sc, lp.read_events(rd))
        closed = [g for g in segs if g["end"] == coll and g["closed_by"] == kind]
        saved = lp.save_segment_final(rd, clean_path(ss.run_name), closed[-1], sc, _eff, fp, int(ss.half_window),
                                      bool(ss.zero_masked), extra_ranges() if ss.use_extra_mask else [], dict(npt=ss.npt, rmin=ss.rmin, rmax=ss.rmax),
                                      tag=kind) if closed else None
    except Exception as e:  # the marker is already written; only the final pattern failed
        saved = None
        ss["ev_msg"] = ("warning", f"Marker saved, but the final pattern of that segment could not be written: {e}")
    if ss.get("ev_msg") is None:
        ss["ev_msg"] = ("success", f"Marked at collection {coll}: {lp.EVENT_LABELS[kind]}." +
                        (f" Final pattern of the segment saved: {Path(saved).name}" if saved else " (No hit patterns in that segment yet, so no pattern was saved.)"))
    ss["ev_note"] = ""
    ss["ev_coll"] = 0
    if kind in ("new_reference", "finished"):
        ss["ev_prompt"] = dict(kind=kind, coll=coll)


def start_new_run():
    """on_click callback: stop the old worker, switch the run name, clear everything that belonged to the old run."""
    ss = st.session_state
    new = clean_path(ss.get("ev_new_run", ""))
    if not new:
        return
    try:
        rd = run_dir()
        if rd is not None:
            (rd / "STOP").write_text("stop", encoding="utf-8")
    except Exception:
        pass
    for k in PER_RUN_STATE:
        ss.pop(k, None)
    ss["run_name"] = new
    ss["ev_new_run"] = ""
    ss["ev_checklist"] = new
    ss["ev_msg"] = None

# ---------------------------------------------------------------------------
# session state (last session's settings are restored, so a restart does not mean retyping)
# ---------------------------------------------------------------------------
PERSIST = ["run_name", "output_dir", "watch_dir", "prefix", "suffix", "frame_exts", "poll_s", "poni_path", "mask_path",
           "calib_hdf_path", "use_extra_mask", "zero_masked", "extra_mask_text", "npt", "rmin", "rmax",
           "half_window", "ml_v1_dir", "ml_override", "ml_lo", "ml_hi", "ref_folder", "view_ref_path"]

if "_settings_loaded" not in st.session_state:
    try:
        if SETTINGS_FILE.is_file():
            for _k, _v in json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).items():
                st.session_state.setdefault(_k, _v)
    except Exception:
        pass
    st.session_state["_settings_loaded"] = True

for k, v in {"poni_path": "", "mask_path": "", "calib_hdf_path": "", "calib_loaded": False,
             "ai": None, "mask": None, "run_name": "", "output_dir": "", "watch_dir": "",
             "prefix": "I11-1-", "suffix": "_subtracted", "frame_exts": "npy,hdf,nxs", "poll_s": 2.0, "ml_v1_dir": "",
             "ml_override": False, "ml_lo": 0.01, "ml_hi": 0.90,
             "use_extra_mask": True, "zero_masked": True, "extra_mask_text": DEFAULT_MASK_TEXT,
             "npt": 2000, "rmin": 1.0, "rmax": 30.0, "half_window": 4}.items():
    st.session_state.setdefault(k, v)

if not st.session_state.calib_loaded and "_calib_autoload_tried" not in st.session_state:
    st.session_state["_calib_autoload_tried"] = True
    try:                                    # restore last session's calibration quietly
        _pp, _mp = clean_path(st.session_state.poni_path), clean_path(st.session_state.mask_path)
        if _pp and os.path.isfile(_pp) and _mp and os.path.isfile(_mp):
            import pyFAI
            st.session_state.ai = pyFAI.load(_pp)
            st.session_state.mask = np.load(_mp).astype(bool)
            st.session_state.calib_loaded = True
            st.session_state["_calib_restored"] = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Run setup")
    st.text_input("Run name", key="run_name", placeholder="Run7_CBZ_seed",
                  help="Names the folder every stage is written into, and the merged patterns.")
    path_input("Output folder", "output_dir", placeholder=r"E:/AP40/2D_output",
               help="Everything is written into <output folder>/<run name>/.")
    path_input("Watched folder (subtracted frames)", "watch_dir", placeholder=r"E:/AP40/subtracted",
               help="The folder the subtraction script writes I11-1-<number>_subtracted.npy into. "
                    "The app and the worker only read it. It must not be the output folder.")
    _out_base = clean_path(st.session_state.output_dir)
    if _out_base:
        st.caption(f"Outputs go to `{Path(_out_base) / (clean_path(st.session_state.run_name) or 'run')}`")
    else:
        st.markdown('<p class="status-warn">● No output folder set</p>', unsafe_allow_html=True)
    if clean_path(st.session_state.watch_dir) and not os.path.isdir(clean_path(st.session_state.watch_dir)):
        st.markdown('<p class="status-warn">● Watched folder not found</p>', unsafe_allow_html=True)
    with st.expander("File name pattern"):
        c1, c2 = st.columns(2)
        c1.text_input("Prefix", key="prefix", help="Not case sensitive. Empty = anything before the number.")
        c2.text_input("Suffix (optional)", key="suffix", help="Frames without it are still found.")
        st.text_input("File types", key="frame_exts", help="npy, hdf and nxs are read. If one frame exists in several types, the first listed is used.")
        st.caption(f"Looks for `{st.session_state.prefix}<collection number>[{st.session_state.suffix}].{{{st.session_state.frame_exts}}}`")

    st.divider()
    st.markdown("### Calibration")
    if st.session_state.calib_loaded:
        st.markdown('<p class="status-ok">● Calibration ready</p>', unsafe_allow_html=True)
        st.caption(f"PONI: `{Path(st.session_state.poni_path).name}`")
        if st.session_state.mask_path:
            st.caption(f"Mask: `{Path(st.session_state.mask_path).name}`")
        if st.session_state.get("_calib_restored"):
            st.caption("Restored from last session. Re-verify in tab 0 if the beamtime or detector changed.")
    else:
        st.markdown('<p class="status-warn">● No calibration loaded</p>', unsafe_allow_html=True)
        st.caption("Go to tab 0 to load and verify it.")

    st.divider()
    st.markdown("### Extra 2θ mask")
    st.checkbox("Apply extra 2θ mask (FEP reflection)", key="use_extra_mask",
                help="Masked pixels are ignored on integration and bridged by a straight line in the 1D pattern. "
                     "The worker is told this when it starts, so restart it after changing it.")
    st.text_input("Range(s), degrees", key="extra_mask_text", help="Several ranges: 5.30-6.10, 8.0-8.4")
    if clean_path(st.session_state.extra_mask_text) != DEFAULT_MASK_TEXT:
        if st.button(f"Reset to {DEFAULT_MASK_TEXT}", key="reset_mask"):
            st.session_state.extra_mask_text = DEFAULT_MASK_TEXT
            st.rerun()
    st.caption(f"Default {DEFAULT_MASK_TEXT}° (set 07/10/26). The carbamazepine 6.10° peak sits on its edge and is lost, "
               "and any peak of another crystal form inside this window is hidden too.")
    try:
        _rng = extra_ranges()
    except ValueError as e:
        _rng = []
        st.error(f"Mask range not understood: {e}")

    with st.expander("Integration and baseline settings"):
        st.number_input("Integration points", 200, 10000, key="npt", step=100,
                        help="Used by the worker when it integrates each frame (restart the worker to change it).")
        c1, c2 = st.columns(2)
        c1.number_input("Min 2θ (°)", 0.0, 60.0, key="rmin", step=0.5)
        c2.number_input("Max 2θ (°)", 1.0, 80.0, key="rmax", step=0.5)
        st.checkbox("Set the masked window to 0 after baselining", key="zero_masked",
                    help="Sets the whole masked window to 0 in the baselined patterns (03, 04). The 02 patterns keep the straight line.")
        st.slider("MOR baseline half_window", 2, 30, key="half_window",
                  help="4 was used for every AP39 test. 5 to 10 is typical for sharp peaks.")

    st.divider()
    st.markdown("### Classifier and worker")
    path_input("Classifier folder (frozen Version 1.2)", "ml_v1_dir", placeholder=_ml_default_dir(),
               help="The frozen classifier folder: beamline_pack_AP40/ML_version1_2. Leave empty to use it automatically if it is found. "
                    "Do not use ML_version1, ML_version1_1, ML_version2_augmented, ML_version3_subtraction_robust or any *_noDLM / *_noGLY folder.")
    _mi = model_info(st.session_state.ml_v1_dir)
    if not _mi["ok"]:
        st.markdown('<p class="status-bad">● Classifier folder not usable</p>', unsafe_allow_html=True)
        st.caption(f"`{_mi['folder']}`  missing: {', '.join(_mi['missing']) or 'record.json unreadable'}")
    elif _mi["version"] != ML_FROZEN_VERSION:
        st.markdown(f'<p class="status-warn">● This is Version {_mi["version"]}, not the frozen {ML_FROZEN_VERSION}</p>', unsafe_allow_html=True)
        st.caption(f"`{_mi['folder']}` · cut-offs {_mi['lower']} / {_mi['upper']}")
    else:
        _cut_ok = (_mi["lower"], _mi["upper"]) == (0.01, 0.9)
        st.markdown(f'<p class="{"status-ok" if _cut_ok else "status-warn"}">● Version {_mi["version"]} (frozen) · {_mi["rows"]} training frames · '
                    f'cut-offs {_mi["lower"]} / {_mi["upper"]}</p>', unsafe_allow_html=True)
        st.caption(f"`{_mi['folder']}`" + ("" if _cut_ok else "  (not the 0.01 / 0.90 freeze values)"))
    st.checkbox("Override the cut-offs by hand", key="ml_override",
                help="Off: the cut-offs come from the model's config.json (0.01 / 0.90 for Version 1.2). "
                     "Whatever the worker used is recorded in worker_manifest.json and in every row of ml_scores.csv.")
    if st.session_state.ml_override:
        o1, o2 = st.columns(2)
        o1.number_input("Lower", 0.0, 1.0, key="ml_lo", step=0.01, format="%.3f")
        o2.number_input("Upper", 0.0, 1.0, key="ml_hi", step=0.01, format="%.3f")
    st.number_input("Folder check interval (s)", 0.5, 30.0, key="poll_s", step=0.5)

    _hb0 = read_heartbeat()
    _alive0 = bool(_hb0 and _hb0.get("status") == "running" and time.time() - float(_hb0.get("updated_epoch", 0)) < max(15.0, 4 * float(st.session_state.poll_s)))
    w1, w2 = st.columns(2)
    start_clicked = w1.button("▶ Start worker", key="worker_start", type="primary", use_container_width=True, disabled=_alive0)
    stop_clicked = w2.button("■ Stop", key="worker_stop", use_container_width=True, disabled=not _alive0)

    if start_clicked:
        _rd = run_dir()
        _errs = []
        if _rd is None:
            _errs.append("set the output folder")
        if not clean_path(st.session_state.run_name):
            _errs.append("set a run name")
        if not (clean_path(st.session_state.watch_dir) and os.path.isdir(clean_path(st.session_state.watch_dir))):
            _errs.append("set an existing watched folder")
        if not (clean_path(st.session_state.poni_path) and os.path.isfile(clean_path(st.session_state.poni_path))):
            _errs.append("load a .poni in tab 0")
        if not (clean_path(st.session_state.mask_path) and os.path.isfile(clean_path(st.session_state.mask_path))):
            _errs.append("load a mask in tab 0")
        _mi2 = model_info(st.session_state.ml_v1_dir)
        if not _mi2["ok"]:
            _errs.append("point the classifier folder at ML_version1_2 (it is missing: " + (", ".join(_mi2["missing"]) or "record.json") + ")")
        if st.session_state.ml_override and st.session_state.ml_lo >= st.session_state.ml_hi:
            _errs.append("make the lower cut-off smaller than the upper")
        if _errs:
            st.error("Before starting the worker: " + "; ".join(_errs) + ".")
        else:
            try:
                _rd.mkdir(parents=True, exist_ok=True)
                _cmd = worker_command(_rd)
                _con = open(_rd / "worker_console.log", "a", encoding="utf-8")
                if os.name == "nt":
                    subprocess.Popen(_cmd, creationflags=0x00000008 | 0x00000200, stdout=_con, stderr=_con, stdin=subprocess.DEVNULL)
                else:
                    subprocess.Popen(_cmd, start_new_session=True, stdout=_con, stderr=_con, stdin=subprocess.DEVNULL)
                st.success("Worker starting. It takes a few seconds to load the classifier.")
                st.code(" ".join(f'"{c}"' if " " in c else c for c in _cmd), language="bash")
            except Exception as e:
                st.error(f"Could not start the worker: {e}. Run the command above by hand in a terminal instead.")
    if stop_clicked:
        try:
            (run_dir() / "STOP").write_text("stop", encoding="utf-8")
            st.info("Stop requested. The worker finishes the frame it is on, saves the running merge and exits.")
        except Exception as e:
            st.error(f"Could not request a stop: {e}")

    @live_fragment(5)
    def sidebar_worker_status():
        hb = read_heartbeat()
        poll = float(st.session_state.poll_s)
        if hb is None:
            st.markdown('<p class="status-warn">● Worker not started</p>', unsafe_allow_html=True)
            return
        age = time.time() - float(hb.get("updated_epoch", 0))
        if hb.get("status") == "stopped":
            st.markdown('<p class="status-warn">● Worker stopped</p>', unsafe_allow_html=True)
        elif age > max(15.0, 4 * poll):
            st.markdown(f'<p class="status-bad">● WORKER NOT RESPONDING (last seen {age:.0f} s ago)</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-ok">● Worker running</p>', unsafe_allow_html=True)
        spf = hb.get("sec_per_frame")
        st.caption(f"{hb.get('processed_total', 0)} frames done · {hb.get('backlog', 0)} waiting"
                   + (f" · {spf:.2f} s per frame" if spf else "")
                   + f" · cut-offs {hb.get('lower')}/{hb.get('upper')}")
        snap = st.session_state.get("sort_snap")
        n_now = len(read_scores_df())
        if snap is not None and n_now > len(snap):
            st.caption(f"{n_now - len(snap)} new frame(s) scored since the sorting list was refreshed.")

    st.divider()
    st.markdown("**Live viewer**")
    st.markdown('<a href="/Live_viewer" target="_blank" rel="noopener">↗ Open the live 1D pattern in a new window</a>', unsafe_allow_html=True)
    st.caption("Cumulative hit pattern with reference overlays. Drag the new window to another screen.")
    with st.expander("Run events: conditions changed / run finished", expanded=bool(st.session_state.get("ev_prompt"))):
        st.radio("What happened?", ["change", "new_reference", "finished"], key="ev_kind", format_func=lambda k: lp.EVENT_LABELS[k],
                 help="Any of these closes the current segment at that collection number and saves its cumulative final pattern. "
                      "The next segment starts at the following frame. A new reference (e.g. a new antisolvent ratio) also means a new run.")
        st.text_input("Note", key="ev_note", placeholder="e.g. antisolvent 20 to 30 %  (free text; details go in your log book)")
        st.number_input("Last collection number of the segment (0 = latest scored)", 0, 10_000_000_000, key="ev_coll", step=1)
        st.button("Mark it", key="ev_mark", on_click=mark_event, use_container_width=True)
        _m = st.session_state.get("ev_msg")
        if _m:
            getattr(st, _m[0])(_m[1])
        _p = st.session_state.get("ev_prompt")
        if _p:
            st.markdown("---")
            st.markdown("**Start a new run?**" if _p["kind"] == "new_reference" else "**Run finished. Name the next run**")
            st.text_input("New run name", key="ev_new_run", placeholder="Run8_CBZ_AS30")
            _hbp = read_heartbeat()
            _alivep = bool(_hbp and _hbp.get("status") == "running" and time.time() - float(_hbp.get("updated_epoch", 0)) < 30)
            _back = int(_hbp.get("backlog", 0)) if _alivep else 0
            if _back:
                st.warning(f"{_back} frame(s) are still waiting for the worker. Wait until it has caught up, so none are lost to the old run.")
            q1, q2 = st.columns(2)
            q1.button("Start new run", key="ev_go", type="primary", on_click=start_new_run, disabled=bool(_back) or not clean_path(st.session_state.get("ev_new_run", "")),
                      use_container_width=True)
            if q2.button("Not now", key="ev_no", use_container_width=True):
                st.session_state.pop("ev_prompt", None)
                st.rerun()
        _cl = st.session_state.get("ev_checklist")
        if _cl:
            st.success(f"Now on run `{_cl}`. The old worker was told to stop.")
            st.markdown("1. **Tab 0:** build the new merged reference (output goes into the new run folder)\n"
                        "2. Give it to the subtraction script and restart that script\n"
                        "3. Set the **watched folder** above if the script now writes somewhere else\n"
                        "4. Press **▶ Start worker** once the old one shows stopped")
            if st.button("Done", key="ev_cl_done"):
                st.session_state.pop("ev_checklist", None)
                st.rerun()

    sidebar_worker_status()
    st.divider()
    st.caption(APP_VERSION)

# remember this session's settings for next time (and keep a copy with the run)
try:
    _snap = {k: st.session_state.get(k) for k in PERSIST if isinstance(st.session_state.get(k), (str, int, float, bool))}
    _js = json.dumps(_snap, sort_keys=True)
    if _js != st.session_state.get("_persist_hash"):
        SETTINGS_FILE.write_text(_js, encoding="utf-8")
        _rd0 = run_dir()
        if _rd0 is not None and clean_path(st.session_state.run_name):
            _rd0.mkdir(parents=True, exist_ok=True)
            (_rd0 / "session_settings.json").write_text(_js, encoding="utf-8")
        st.session_state["_persist_hash"] = _js
except Exception:
    pass

st.title("2D PXRD · inline analysis")
(tab_start, tab_watch, tab_sort, tab_int, tab_stack) = st.tabs([
    "0 · Start-up", "1 · Watcher", "2 · Visualiser + sorting", "3 · Integrate + baseline", "4 · Pattern stacker"])
tab_cal = tab_ref = tab_start



# ===========================================================================
# TAB 0: CALIBRATION   (steps and wording follow the 1D app)
# ===========================================================================
def load_calib_image(path):
    import h5py
    with h5py.File(path, "r") as f:
        for p in ["entry/data/data", "entry1/data/data", "entry1/pixium_hdf/data",
                  "entry1/Pixium10:detector/data", "entry/detector/data"]:
            if p in f:
                d = f[p][()]
                return (d[0] if d.ndim == 3 else d).astype(np.float32)
        found = []
        f.visititems(lambda n, o: found.append(n) if hasattr(o, "ndim") and o.ndim in (2, 3) else None)
        if found:
            d = f[found[0]][()]
            return (d[0] if d.ndim == 3 else d).astype(np.float32)
    raise RuntimeError(f"No 2D dataset found in {path}")


with tab_cal:
    st.markdown("### Calibration · start of beamtime")
    st.caption("Do this once per beamtime window. Load the calibration image, run pyFAI to save a "
               ".poni and a mask, then load and verify them here. Every other tab uses them.")

    st.markdown("#### Step 1 · Load calibration image")
    calib_hdf = path_input("Path to calibration .hdf file", "calib_hdf_path", mode="file",
                           placeholder=r"E:/calib/pixium_122554.hdf",
                           filetypes=[("HDF/NXS files", "*.hdf *.nxs"), ("All files", "*.*")],
                           help="The CeO2 (or equivalent) calibration frame collected at the beamline.")
    if st.button("Load image", key="load_calib_img"):
        if not os.path.isfile(calib_hdf):
            st.error(f"File not found:\n{calib_hdf}")
        else:
            try:
                img = load_calib_image(calib_hdf)
                st.success(f"Loaded · shape {img.shape[0]} × {img.shape[1]} px")
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99), origin="lower")
                ax.set_title("Calibration image preview", fontsize=10); ax.axis("off")
                fig.tight_layout(); st.pyplot(fig, use_container_width=False); plt.close(fig)
            except Exception as e:
                st.error(f"Could not load image: {e}")

    st.divider()
    st.markdown("#### Step 2 · Run pyFAI calibration")
    st.caption("Opens the pyFAI calibration GUI in a separate window. Set the wavelength, pick the "
               "calibrant, mask bad pixels, pick rings, fit, then save the .poni and export the mask "
               "as .npy before closing.")
    cA, cB = st.columns([1, 2])
    launch = cA.button("Launch pyFAI calibration GUI →", type="primary", key="launch_pyfai",
                       disabled=not bool(calib_hdf), use_container_width=True,
                       help="Give the calibration image in Step 1 first.")
    cB.info("pyFAI opens in a separate window. Finish there, save your files, then come back to Step 3.")
    if launch:
        import subprocess
        pdir = Path(sys.executable).parent
        exe = next((str(p) for p in [pdir / "Scripts" / "pyFAI-calib2.exe", pdir / "Scripts" / "pyfai-calib2.exe",
                                     pdir / "pyFAI-calib2.exe"] if p.exists()), None)
        cmd = [exe, calib_hdf] if exe else [sys.executable, "-m", "pyFAI.app.calib2", calib_hdf]
        st.markdown("**Command to run manually if needed:**")
        st.code(" ".join(f'"{c}"' if " " in c else c for c in cmd), language="bash")
        try:
            if os.name == "nt":
                subprocess.Popen(cmd, creationflags=0x00000008 | 0x00000200, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            st.success("pyFAI-calib2 launch requested. It can take 15 to 30 seconds to open. "
                       "If nothing appears, use the manual command above.")
        except Exception as e:
            st.error(f"Auto-launch failed: {e}. Use the manual command above instead.")

    st.divider()
    st.markdown("#### Step 3 · Load and verify the saved calibration")
    st.caption("Point to the .poni and mask saved by pyFAI. A test integration of the calibration "
               "image confirms they work before you start.")
    col3a, col3b = st.columns(2)
    with col3a:
        poni_in = path_input("Path to saved .poni file", "poni_path", mode="file",
                             placeholder=r"E:/calib/calib.poni",
                             filetypes=[("PONI files", "*.poni"), ("All files", "*.*")])
    with col3b:
        mask_in = path_input("Path to saved mask (.npy, True = bad pixel)", "mask_path", mode="file",
                             placeholder=r"E:/calib/mask.npy",
                             filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])

    if st.button("Load and verify calibration", type="primary", key="verify_calib"):
        ok = True
        if not os.path.isfile(poni_in):
            st.error(f".poni file not found:\n{poni_in}"); ok = False
        if mask_in and not os.path.isfile(mask_in):
            st.error(f"Mask file not found:\n{mask_in}"); ok = False
        if ok:
            try:
                import pyFAI
                ai = pyFAI.load(poni_in)
                mask = np.load(mask_in).astype(bool) if mask_in else None
                st.session_state.ai, st.session_state.mask = ai, mask
                st.session_state.calib_loaded = True
                st.success("Calibration loaded and stored. All tabs are ready.")
                st.markdown(f"""
| Parameter | Value |
|---|---|
| Distance (m) | `{ai.dist:.5f}` |
| poni1 (m) | `{ai.poni1:.5f}` |
| poni2 (m) | `{ai.poni2:.5f}` |
| Wavelength (Å) | `{ai.wavelength * 1e10:.5f}` |
| Pixel size (μm) | `{ai.pixel1 * 1e6:.1f} × {ai.pixel2 * 1e6:.1f}` |
""")
                if mask is not None:
                    st.caption(f"Mask shape {mask.shape[1]} × {mask.shape[0]} px, "
                               f"{100 * mask.mean():.2f}% of pixels masked.")
                if calib_hdf and os.path.isfile(calib_hdf):
                    with st.spinner("Integrating the calibration image..."):
                        img = load_calib_image(calib_hdf)
                        if mask is not None and mask.shape != img.shape:
                            st.error(f"The mask is {mask.shape} but the calibration image is {img.shape}.")
                        else:
                            x_c, y_c = fp.integrate(img, ai, mask, 1500, 1.0, 30.0)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(x_c, y_c, color="#2c3e50", linewidth=0.8)
                            ax.set_xlabel("2θ (°)"); ax.set_ylabel("Intensity"); ax.set_xlim(1, 30)
                            ax.set_title("Test integration · calibrant peaks should be sharp and in the right places")
                            ax.grid(alpha=0.2); fig.tight_layout(); _pyplot(fig)
                            st.caption("If peaks look broad, off-position or missing, re-run the pyFAI "
                                       "calibration and check the ring fitting.")
                else:
                    st.info("Give a calibration image in Step 1 to also run a test integration.")
            except Exception as e:
                st.session_state.calib_loaded = False
                st.error(f"Failed to load calibration: {e}")



# ===========================================================================
# TAB 1: BUILD REFERENCE
# ===========================================================================
def _view_reference():
    """Draw the reference stored in st.session_state['ref_view'] (works in a pop-up or inline)."""
    rv = st.session_state.get("ref_view")
    if not rv:
        return
    mean, var, tmask = rv["mean"], rv["var"], rv["mask"]
    valid = ~tmask if tmask is not None else np.ones(mean.shape, bool)
    st.markdown(f"**{rv['label']}** · {rv['n_used']} frames · {mean.shape[1]} × {mean.shape[0]} px · "
                f"mean level {mean[valid].mean():.1f} counts")
    lo, hi = np.percentile(mean[valid], [1, 99.5])
    c1, c2 = st.columns(2)
    vmin = c1.number_input("Colour scale min", value=float(round(lo)), key="rv_vmin")
    vmax = c2.number_input("Colour scale max", value=float(round(hi)), key="rv_vmax")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    show_image(ax[0], mean, vmin, vmax, "Merged reference (grey = masked)", "viridis", tmask)
    snr = mean[valid] / np.sqrt(var[valid] + 1e-12)
    ax[1].hist(snr, bins=100, log=True)
    ax[1].set_xlabel("per-pixel signal-to-noise (mean / standard error)"); ax[1].set_title("Reference noise")
    plt.tight_layout(); _pyplot(fig)
    if rv.get("profile") is not None:
        x, y = rv["profile"]
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x, y, lw=0.8); ax.set_xlabel("2θ (°)"); ax.set_ylabel("Counts")
        ax.set_title("Reference integrated to 1D (the FEP line and the humps should look sensible)")
        ax.grid(alpha=0.2); plt.tight_layout(); _pyplot(fig)


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
if _dialog:
    @_dialog("Reference frame", width="large")
    def reference_popup():
        _view_reference()
else:
    def reference_popup():
        with st.expander("Reference frame", expanded=True):
            _view_reference()


def set_ref_view(ref, label=None):
    """Remember a reference for the pop-up. Adds a 1D profile if the calibration is loaded."""
    mask = None
    prof = None
    try:
        ai, det = calib()
        mask = fp.total_mask(det, ai, ref.mean.shape, extra_ranges())
        prof = fp.integrate(ref.mean, ai, mask, int(st.session_state.npt), float(st.session_state.rmin),
                            float(st.session_state.rmax))
    except Exception:
        det = st.session_state.mask
        mask = det if (det is not None and det.shape == ref.mean.shape) else None
    st.session_state["ref_view"] = dict(mean=ref.mean, var=ref.var_of_mean, mask=mask, profile=prof,
                                        label=label or ref.label, n_used=ref.n_used)


with tab_ref:
    st.markdown("Merge every 2D reference frame in a folder into **one reference frame** (per-pixel mean "
                "and its variance). Use frames of the solvent in the FEP tube with no crystals, for "
                "example the 70:30 reference frames collected at the start of a run.")
    c1, c2 = st.columns([3, 1])
    with c1:
        ref_folder = path_input("Folder of 2D reference frames", "ref_folder",
                                placeholder=r"E:/AP39/Run_seeded/reference_frames/")
    with c2:
        ref_ext_choice = st.radio("File type", [".hdf", ".nxs", "both (.nxs + .hdf)"], key="ref_ext",
                                  help="Use one type. 'Both' counts every frame twice when each .nxs "
                                       "links to a .hdf, which gives a duplicated reference.")
    ref_recursive = st.checkbox("Include sub-folders", value=False, key="ref_recursive")

    ref_files = []
    if ref_folder:
        try:
            ref_files = fm.find_frames(ref_folder, exts_from_choice(ref_ext_choice), ref_recursive)
            if ref_files:
                st.info(f"{len(ref_files)} frame files found "
                        f"(first: {Path(ref_files[0]).name}, last: {Path(ref_files[-1]).name}).")
            else:
                st.warning("No matching files in that folder.")
        except FileNotFoundError as e:
            st.error(str(e))

    c3, c4 = st.columns(2)
    ref_label = c3.text_input("Reference label (stored in the file and in every corrected frame)",
                              value="70:30 EtOH:H2O merged", key="ref_label")
    ref_out = clean_path(c4.text_input("Save reference as (blank = <output>/<run>/00_reference/merged_reference.npz)",
                                       key="ref_out"))
    screen_on = st.checkbox("Screen out bad frames before merging", value=True, key="screen_on",
                            help="Rejects shutter-closed frames, total-count outliers and frames whose "
                                 "pattern does not match the rest (detector out of position).")
    with st.expander("Screening thresholds"):
        s1, s2, s3 = st.columns(3)
        min_total_frac = s1.number_input("Minimum total counts, fraction of median", 0.0, 1.0, 0.30, 0.05, key="min_total_frac")
        z_total = s2.number_input("Total-count outlier limit (robust σ)", 1.0, 20.0, 5.0, 0.5, key="z_total")
        min_corr = s3.number_input("Minimum pattern correlation with the median frame", 0.0, 1.0, 0.97, 0.01, key="min_corr")

    if st.button("Build merged reference", type="primary", disabled=not ref_files, key="build_ref_btn"):
        try:
            det = st.session_state.mask
            out_path = ref_out or str(stage_dir("00_reference") / "merged_reference.npz") \
                if (ref_out or stage_dir("00_reference")) else str(Path(ref_folder) / "merged_reference.npz")
            if not out_path.lower().endswith(".npz"):
                out_path += ".npz"
            bar = st.progress(0.0, text="Starting...")
            t0 = time.time()

            def _prog(stage, done, total):
                frac = min(1.0, done / max(total, 1))
                bar.progress(0.5 * frac if stage == "screening" else 0.5 + 0.5 * frac, text=f"{stage}: {done}/{total}")

            kw = dict(min_total_frac=min_total_frac, z_total=z_total, min_corr=min_corr) if screen_on else {}
            ref = fm.build_reference(ref_files, mask=det, label=ref_label, screen=screen_on, progress=_prog, **kw)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fm.save_reference(ref, out_path)
            bar.progress(1.0, text=f"Done in {time.time() - t0:.0f} s")
            st.session_state["ref_use"] = out_path
            st.session_state["ref_result"] = dict(path=out_path, n_used=ref.n_used, n_rejected=ref.n_rejected,
                                                  rejected=ref.rejected, shape=ref.mean.shape,
                                                  warnings=fp.check_reference(ref))
            set_ref_view(ref)
        except Exception as e:
            st.error(f"Could not build the reference: {e}")

    res = st.session_state.get("ref_result")
    if res:
        st.success(f"Saved merged reference to {res['path']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames merged", res["n_used"]); m2.metric("Frames rejected", res["n_rejected"])
        m3.metric("Frame size (px)", f"{res['shape'][1]} × {res['shape'][0]}")
        for w in res.get("warnings", []):
            st.warning(w)
        if res["rejected"]:
            st.markdown("**Rejected frames**")
            _df(pd.DataFrame([(Path(p).name, i, why) for p, i, why in res["rejected"]], columns=["file", "frame", "reason"]))
        st.caption("Use this .npz as the reference for the inline subtraction script.")

    st.markdown("#### View a reference frame")
    st.caption("Opens a pop-up with the reference image, its noise and its 1D profile, so you can check "
               "it by eye before using it. Works on the reference just built or on a saved .npz.")
    v1, v2 = st.columns([3, 1])
    view_path = path_input("Saved reference (.npz) to view (blank = the one just built)", "view_ref_path",
                           mode="file", placeholder=r"E:/AP39/reference/merged_reference.npz",
                           filetypes=[("Reference", "*.npz"), ("All files", "*.*")])
    if st.button("🔍 View reference frame", key="view_ref_btn",
                 disabled=not (view_path or st.session_state.get("ref_view"))):
        try:
            if view_path:
                r_ = get_ref(view_path)
                set_ref_view(r_, label=f"{r_.label}  ({Path(view_path).name})")
                for w in fp.check_reference(r_):
                    st.warning(w)
            reference_popup()
        except Exception as e:
            st.error(f"Could not show the reference: {e}")





# ---------------------------------------------------------------------------
# TAB 0 (continued): HOW MANY REFERENCE FRAMES DO I NEED?
# ---------------------------------------------------------------------------
with tab_start:
    st.divider()
    with st.expander("How many reference frames do I need? (reference-size test)"):
        st.caption("Builds the reference from the first N frames, for several values of N, and subtracts each from a few "
                   "frames that are NOT in the reference (raw frames with no crystals). The leftover noise in the subtracted "
                   "frames should fall as N grows, then flatten. The point where it flattens is the number of reference frames "
                   "worth collecting. Use raw frames from the same run conditions, none of them from the reference folder.")
        rt1, rt2 = st.columns([3, 1])
        with rt1:
            rs_ref_folder = path_input("Folder of reference frames (the longest set you collected, e.g. 100 frames)", "rs_ref_folder",
                                       placeholder=r"E:/AP39/Run_seeded/reference_frames/")
            rs_test_folder = path_input("Folder of raw test frames (non-hits, not in the reference)", "rs_test_folder",
                                        placeholder=r"E:/AP39/Run_seeded/raw_nonhit_frames/")
        with rt2:
            rs_ext = st.radio("File type", [".hdf", ".nxs", "both (.nxs + .hdf)"], key="rs_ext")
        rt3, rt4 = st.columns(2)
        rs_points = rt3.text_input("Values of N to test", value="10, 20, 30, 46, 60, 80, 100", key="rs_points")
        rs_ntest = rt4.number_input("Number of test frames", 1, 50, 8, 1, key="rs_ntest")
        rs_files, rs_tests = [], []
        if rs_ref_folder and os.path.isdir(rs_ref_folder):
            rs_files = fm.find_frames(rs_ref_folder, exts_from_choice(rs_ext), False)
        if rs_test_folder and os.path.isdir(rs_test_folder):
            rs_tests = fm.find_frames(rs_test_folder, exts_from_choice(rs_ext), False)[: int(rs_ntest)]
        if rs_files or rs_tests:
            st.caption(f"{len(rs_files)} reference frames and {len(rs_tests)} test frames found. Frames are taken in file-name order.")
        try:
            _ns = sorted({int(v) for v in re.split(r"[,\s]+", rs_points.strip()) if v})
        except ValueError:
            _ns = []
            st.error("Values of N must be whole numbers separated by commas.")
        _ns = [n for n in _ns if 2 <= n <= len(rs_files)]
        if _ns:
            st.caption(f"Will build {len(_ns)} references, loading {sum(_ns)} frames in total. Expect a few minutes.")
        if st.button("Run the reference-size test", disabled=not (_ns and rs_tests and st.session_state.calib_loaded), key="rs_run"):
            try:
                ai_, det_ = calib()
                tests_ = [next(iter(fm.iter_frames([p_])))[2] for p_ in rs_tests]
                tmask_ = fp.total_mask(det_, ai_, tests_[0].shape, extra_ranges())
                valid_ = ~tmask_
                bar = st.progress(0.0, text="Starting...")
                rows_, prev_mean = [], None
                for j, n in enumerate(_ns):
                    bar.progress(j / len(_ns), text=f"Building the reference from the first {n} frames ({j + 1}/{len(_ns)})...")
                    ref_ = fm.build_reference(rs_files[:n], mask=det_, label=f"first {n}", screen=True,
                                              progress=lambda stage, done, total: None)
                    sig_, sc_ = [], []
                    for fr_ in tests_:
                        corr_, info_ = fm.subtract_frame(fr_, ref_, tmask_)
                        v_ = corr_[valid_]
                        sig_.append(1.4826 * float(np.median(np.abs(v_ - np.median(v_)))))
                        sc_.append(float(info_["scale"]))
                    if prev_mean is None:
                        step_ = np.nan
                    else:
                        step_ = float(np.sqrt(np.mean((ref_.mean[valid_] - prev_mean[valid_]) ** 2)))
                    prev_mean = ref_.mean
                    rows_.append(dict(N=n, frames_used=int(ref_.n_used), noise=float(np.mean(sig_)), noise_spread=float(np.std(sig_)),
                                      median_scale=float(np.median(sc_)), change_from_previous=step_))
                bar.progress(1.0, text="Done")
                dfr = pd.DataFrame(rows_)
                _p = stage_dir("00_reference")
                if _p is not None:
                    _p.mkdir(parents=True, exist_ok=True)
                    dfr.to_csv(_p / "reference_size_test.csv", index=False)
                st.session_state["rs_result"] = dfr
            except Exception as e:
                st.error(f"Reference-size test failed: {e}")
        _r = st.session_state.get("rs_result")
        if _r is not None and len(_r):
            floor_ = float(_r["noise"].iloc[-1])
            ok_ = _r[_r["noise"] <= 1.02 * floor_]
            fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
            ax[0].errorbar(_r["N"], _r["noise"], yerr=_r["noise_spread"], fmt="o-", ms=4, capsize=3)
            ax[0].set_xlabel("frames in the reference (N)"); ax[0].set_ylabel("noise left in subtracted frames (counts)")
            ax[0].set_title("Leftover noise vs reference size"); ax[0].grid(alpha=0.3)
            ax[1].plot(_r["N"], _r["change_from_previous"], "s-", ms=4, color="#8e44ad")
            ax[1].set_xlabel("frames in the reference (N)"); ax[1].set_ylabel("RMS change from the previous reference (counts)")
            ax[1].set_title("How much the reference itself still changes"); ax[1].grid(alpha=0.3)
            plt.tight_layout(); _pyplot(fig)
            if len(ok_):
                st.success(f"The leftover noise is within 2% of its value at N = {int(_r['N'].iloc[-1])} from N = {int(ok_['N'].iloc[0])} onwards. "
                           "That is a guide, not a rule: judge the curve and the classifier results on your own data.")
            _df(_r)
            st.caption("Noise = robust spread (1.4826 × median absolute deviation) of the unmasked pixels of the subtracted frames, "
                       "averaged over the test frames. Test frames must not be part of the reference.")


# ===========================================================================
# TAB 1: WATCHER   (a light read-only view; the worker does the real work)
# ===========================================================================
LABEL_COLOURS = {"hit": "#1a7a4a", "non_hit": "#7f8c8d", "unsure": "#e67e22", "error": "#c0392b"}

with tab_watch:
    st.markdown("The **worker** finds each new frame in the watched folder, waits until it is completely written, classifies it, "
                "integrates it and logs it. This page only reads the worker's small log files and refreshes itself every "
                "5 seconds, without touching the other tabs. Start and stop the worker from the sidebar.")

    @live_fragment(5)
    def watch_panel():
        hb = read_heartbeat()
        df = read_scores_df()
        pat = frame_pat()
        wdir = clean_path(st.session_state.watch_dir)
        n_in = 0
        if wdir and os.path.isdir(wdir):
            try:
                n_in = sum(1 for e in os.scandir(wdir) if pat.match(e.name))
            except OSError:
                n_in = 0
        lab_counts = df["ml_label"].value_counts().to_dict() if len(df) else {}
        waiting = (hb or {}).get("backlog", max(n_in - len(df), 0))
        m = st.columns(6)
        m[0].metric("Frames in folder", n_in)
        m[1].metric("Classified", len(df))
        m[2].metric("Hit", lab_counts.get("hit", 0))
        m[3].metric("Unsure", lab_counts.get("unsure", 0) + lab_counts.get("error", 0))
        m[4].metric("Non-hit", lab_counts.get("non_hit", 0))
        m[5].metric("Waiting", waiting)
        if hb:
            age = time.time() - float(hb.get("updated_epoch", 0))
            spf = hb.get("sec_per_frame")
            st.caption(f"Worker {hb.get('status')} · last heard from {age:.0f} s ago · "
                       + (f"{spf:.2f} s per frame · " if spf else "")
                       + f"cut-offs {hb.get('lower')} / {hb.get('upper')} · model {hb.get('model')} {hb.get('model_version')} · "
                       f"{hb.get('hits_in_live_merge', 0)} hits in the running merge · {hb.get('errors', 0)} frame error(s)")
            if hb.get("status") == "running" and age > max(15.0, 4 * float(st.session_state.poll_s)):
                st.error("The worker has stopped responding. Restart it from the sidebar; it carries on where it left off.")
            if waiting and hb.get("sec_per_frame") and float(hb["sec_per_frame"]) > 5:
                st.warning("Frames are taking longer than 5 s each to process, so a backlog may build up.")
        else:
            st.info("The worker has not been started for this run (or the run name and output folder are not set).")
        if len(df):
            d = df.tail(300)
            fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            for lab, col in LABEL_COLOURS.items():
                s = d[d["ml_label"] == lab]
                if len(s):
                    ax[0].scatter(s["collection"], s["score"], s=14, c=col, label=lab)
            lo_, hi_ = df["lower"].iloc[-1], df["upper"].iloc[-1]
            if np.isfinite(lo_) and np.isfinite(hi_):
                ax[0].axhline(lo_, color="#999", ls="--", lw=0.8); ax[0].axhline(hi_, color="#999", ls="--", lw=0.8)
            _ev = lp.read_events(run_dir())
            for _e in _ev.itertuples():
                if not (d["collection"].min() <= _e.collection <= d["collection"].max()):
                    continue
                _c = {"change": "#e67e22", "new_reference": "#c0392b", "finished": "#2c3e50"}.get(_e.event, "#888")
                for _a in ax:
                    _a.axvline(_e.collection, color=_c, lw=1.2, ls="-.")
            ax[0].set_ylabel("hit score"); ax[0].legend(fontsize=8, ncol=4, loc="upper left"); ax[0].grid(alpha=0.2)
            ax[0].set_title("Classifier score per frame (last 300; dashed = cut-offs)", fontsize=10)
            ishit = (d["ml_label"] == "hit").astype(float)
            ax[1].plot(d["collection"], ishit.rolling(20, min_periods=1).mean(), color="#1a7a4a")
            ax[1].set_ylim(0, 1.02); ax[1].set_ylabel("hit fraction"); ax[1].set_xlabel("collection number"); ax[1].grid(alpha=0.2)
            ax[1].set_title("Hit fraction over the last 20 frames", fontsize=10)
            plt.tight_layout(); _pyplot(fig)
            if len(_ev):
                st.markdown("**Run events** (dash-dot lines on the plots)")
                _df(_ev[["time", "collection", "event", "note"]], height=min(35 * len(_ev) + 40, 220))
            st.markdown("**Newest frames**")
            _df(df.tail(12).iloc[::-1][["file", "score", "ml_label", "guard", "sec", "saved"]], height=250)
        rd = run_dir()
        if rd is not None and (rd / "worker.log").is_file():
            try:
                lines = (rd / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                with st.expander("Worker log (last 12 lines)"):
                    st.code("\n".join(lines))
            except OSError:
                pass

    watch_panel()



# ===========================================================================
# TAB 2: VISUALISER + SORTING   (ML labels from the worker; you review the unsure frames by eye)
# ===========================================================================
with tab_sort:
    st.markdown("The classifier labels every frame as it arrives. Frames it was **unsure** about (or that the guard flagged) wait in "
                "the queue below for you to decide by eye. **Your label always wins** over the classifier's, every click is saved "
                "at once, and nothing is moved or deleted. The list only changes when you press Refresh, so a frame never "
                "shifts under you while you are judging it.")
    watch_dir_ = clean_path(st.session_state.watch_dir)
    cur_df = read_scores_df()

    # decisions on screen: loaded from the csv the first time for this run, then kept
    _dkey = (str(sort_csv_path()),)
    if st.session_state.get("_sort_loaded_for") != _dkey:
        st.session_state["sort_dec"] = read_decisions_csv(sort_csv_path())
        st.session_state["sort_snap"] = None
        st.session_state["sort_cur"] = None
        st.session_state["_sort_loaded_for"] = _dkey
        if st.session_state["sort_dec"]:
            st.toast(f"Loaded {len(st.session_state['sort_dec'])} saved decisions")
    dec = st.session_state["sort_dec"]

    rf1, rf2 = st.columns([1, 4])
    refresh = rf1.button("🔄 Refresh frame list", key="sort_refresh", type="primary", use_container_width=True)
    if refresh or st.session_state.get("sort_snap") is None:
        st.session_state["sort_snap"] = cur_df
    snap = st.session_state["sort_snap"]
    n_new = len(cur_df) - len(snap)
    rf2.caption(f"{len(snap)} frames in this list" + (f" · **{n_new} new frame(s) scored since** (press Refresh to add them)" if n_new > 0 else ""))

    names = snap["file"].astype(str).tolist()
    ml = {str(r.file): (float(r.score) if r.score == r.score else float("nan"), str(r.ml_label), str(r.guard)) for r in snap.itertuples()}

    def eff(n):
        return dec.get(n) or ml[n][1]

    counts = {k: sum(1 for n in names if eff(n) == k) for k in ("hit", "non_hit", "unsure", "bad", "error")}
    queue_all = [n for n in names if ml[n][1] in ("unsure", "error") and n not in dec]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Frames", len(names)); m2.metric("Hit", counts["hit"]); m3.metric("Non-hit", counts["non_hit"])
    m4.metric("Unsure (still marked)", counts["unsure"] + counts["error"]); m5.metric("Bad frame", counts["bad"])
    m6.metric("Waiting for your decision", len(queue_all))
    if sort_csv_path() is None:
        st.warning("No output folder or run name is set, so decisions are kept on screen only.")

    with st.expander("Run-start check (the first frames of a run should be empty)", expanded=False):
        st.caption("Checks the first frames of the run, which should be the empty reference frames, before you trust the sorting: "
                   "frame sanity, how many score as hits, and whether the empty pixels match the calibration mask. "
                   "It only warns; it never changes a label.")
        rs_n = st.number_input("Number of first frames to check", 3, 30, 10, 1, key="rs_n")
        _mp = clean_path(st.session_state.mask_path) if st.session_state.mask_path else ""
        if st.button("Run the run-start check", disabled=not (names and watch_dir_), key="rs_run_chk"):
            try:
                _mlf = clean_path(st.session_state.ml_v1_dir) or _ml_default_dir()
                _clf = load_ml_v1(_mlf)
                _rs, _fg = _ml_mods(_mlf)
                cal_ = _rs.calibration_map(_mp) if _mp else None
                frs_ = [load_npy_cached(str(Path(watch_dir_) / n_), os.path.getmtime(Path(watch_dir_) / n_)) for n_ in names[:int(rs_n)]]
                res_ = _rs.check_run_start(frs_, _clf, calibration=cal_)
                (st.success if res_["status"] == "OK" else st.warning if res_["status"] == "WARN" else st.error)(f"Run-start check: {res_['status']}")
                for why_ in res_["reasons"]:
                    st.write("• " + why_)
                st.caption(f"Scores of the first {res_['n']} frames: {res_['scores']}"
                           + (f" · calibration mask coverage {res_['cov_median']:.3f}" if res_.get("cov_median") is not None else ""))
                _sd = stage_dir("01b_sorting")
                if _sd is not None:
                    _sd.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame([dict(checked=datetime.datetime.now().isoformat(timespec="seconds"), status=res_["status"],
                                       n=res_["n"], reasons=" | ".join(res_["reasons"]), mask=_mp)]).to_csv(
                        _sd / "run_start_check.csv", mode="a", header=not (_sd / "run_start_check.csv").is_file(), index=False)
            except Exception as e:
                st.error(f"Run-start check could not run: {e}")

    st.markdown("##### Review")
    show = st.radio("Show", ["Unsure queue (not yet decided)", "All frames, newest first",
                             "Hits, lowest score first (look for false hits)", "Frames I have labelled"],
                    key="sort_show", horizontal=True)
    view_kind = st.radio("Image", ["Corrected frame", "Spot view (ring median removed)"], key="sort_view", horizontal=True,
                         help="The spot view removes each ring's median so isolated spots stand out. A smooth powder ring is removed with it, "
                              "so judge those from the pattern on the right.")
    if show.startswith("Unsure"):
        order = queue_all
    elif show.startswith("All"):
        order = names[::-1]
    elif show.startswith("Hits"):
        order = sorted([n for n in names if eff(n) == "hit"], key=lambda n: (ml[n][0] != ml[n][0], ml[n][0]))
    else:
        order = [n for n in names if n in dec]
    st.session_state["_sort_order"] = order

    def _goto(name):
        st.session_state["sort_cur"] = name

    def _label(name, lbl):
        o = st.session_state.get("_sort_order", [])
        st.session_state["sort_dec"][name] = lbl
        save_decisions(st.session_state["sort_dec"], clean_path(st.session_state.watch_dir))
        if name in o:
            i = o.index(name)
            st.session_state["sort_cur"] = o[i + 1] if i + 1 < len(o) else (o[i - 1] if i > 0 else name)

    def _clear(name):
        st.session_state["sort_dec"].pop(name, None)
        save_decisions(st.session_state["sort_dec"], clean_path(st.session_state.watch_dir))

    if not order:
        if show.startswith("Unsure") and names:
            st.success("No unsure frames are waiting. Press Refresh as new frames arrive.")
        elif not names:
            st.info("No frames have been classified yet. Start the worker in the sidebar and check the Watcher tab.")
        else:
            st.info("No frames to show in this view.")
    else:
        cur = st.session_state.get("sort_cur")
        if cur not in order:
            cur = order[0]
        st.session_state["sort_cur"] = cur
        i_ = order.index(cur)
        sc_ml, lab_ml, guard_ml = ml[cur]
        lab = dec.get(cur)
        tag = {"hit": "🟢 Hit", "non_hit": "⚪ Non-hit", "unsure": "🟠 Unsure", "bad": "🔴 Bad frame"}.get(lab, "⬜ not decided by you")
        st.markdown(f"**{cur}**  ·  position {i_ + 1} of {len(order)}  ·  {tag}")
        st.caption(f"ML score {sc_ml:.3f} (probability of a hit) · ML label: {SORT_LABELS.get(lab_ml, lab_ml)}"
                   + (f" · guard flag: {guard_ml}" if guard_ml else "")
                   + (" · you changed this" if lab not in (None, lab_ml) else ""))
        b1, b2, b3, b4, b7, b5, b6 = st.columns(7)
        b1.button("◀ Back", key="sort_back", on_click=_goto, args=(order[max(i_ - 1, 0)],), disabled=i_ == 0, use_container_width=True)
        b2.button("🟢 Hit", key="sort_hit", on_click=_label, args=(cur, "hit"), type="primary", use_container_width=True)
        b3.button("⚪ Non-hit", key="sort_non", on_click=_label, args=(cur, "non_hit"), use_container_width=True)
        b4.button("🟠 Unsure", key="sort_uns", on_click=_label, args=(cur, "unsure"), use_container_width=True)
        b7.button("🔴 Bad frame", key="sort_bad", on_click=_label, args=(cur, "bad"), use_container_width=True)
        b5.button("Skip ▶", key="sort_skip", on_click=_goto, args=(order[min(i_ + 1, len(order) - 1)],), disabled=i_ >= len(order) - 1, use_container_width=True)
        b6.button("Clear my label", key="sort_clear", on_click=_clear, args=(cur,), disabled=lab is None, use_container_width=True)

        pth = Path(watch_dir_) / cur if watch_dir_ else None
        if pth is None or not pth.is_file():
            st.warning("The frame file is not in the watched folder any more (or the folder is not set).")
        else:
            try:
                fr_ = load_npy_cached(str(pth), os.path.getmtime(pth))
                rk = tuple(extra_ranges())
                ai_ = det_ = None
                if st.session_state.calib_loaded:
                    ai_, det_ = calib()
                    tm_ = fp.total_mask(det_, ai_, fr_.shape, list(rk))
                else:
                    tm_ = fr_ == 0
                    st.info("Load the calibration in tab 0 for the exact mask; pixels equal to 0 are shown as masked until then.")
                valid_ = ~tm_
                cA, cB = st.columns([1, 1.25])
                with cA:
                    if view_kind.startswith("Spot"):
                        rm_ = _ring_map_cached(_geom_key(ai_, tm_), rk, ai_, tm_) if ai_ is not None else None
                        if rm_ is None:
                            st.info("The spot view needs the calibration (tab 0).")
                            img_, lo_, hi_, cm_, ttl_ = np.where(tm_, np.nan, fr_)[::2, ::2], *np.percentile(fr_[valid_], [1, 99.5]), "gray", "Corrected frame (masked = blue)"
                        else:
                            hp_ = fm.ring_median_subtract(fr_, rm_)
                            mad_ = 1.4826 * np.median(np.abs(hp_[valid_] - np.median(hp_[valid_]))) + 1e-9
                            img_ = block_max(np.where(tm_, 0, hp_ / mad_), 2)
                            bm_ = ~block_max((~tm_).astype(np.uint8), 2).astype(bool)
                            img_ = np.where(bm_, np.nan, img_)
                            lo_, hi_, cm_, ttl_ = 0, 12, "magma", "Spot view in noise units (masked = blue)"
                    else:
                        lo_, hi_ = np.percentile(fr_[valid_], [1, 99.5])
                        ttl_, cm_ = "Corrected frame (masked = blue)", "gray"
                        img_ = np.where(tm_, np.nan, fr_)[::2, ::2]
                    cmap_ = plt.get_cmap(cm_).copy()
                    cmap_.set_bad(MASK_BLUE)
                    fig, ax = plt.subplots(figsize=(5.2, 5.2))
                    ax.imshow(img_, origin="lower", vmin=lo_, vmax=hi_, cmap=cmap_)
                    ax.set_title(ttl_, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                    plt.tight_layout(); _pyplot(fig)
                with cB:
                    xy_ = xy_for(cur)
                    if xy_ is not None:
                        x_, y_ = fp.load_xy(str(xy_))
                    elif ai_ is not None:
                        x_, y_ = fp.integrate(fr_, ai_, tm_, int(st.session_state.npt), float(st.session_state.rmin), float(st.session_state.rmax))
                        y_ = fp.fill_gaps(x_, y_, list(rk))
                    else:
                        x_ = y_ = None
                    if x_ is None:
                        st.info("No stored pattern for this frame yet and no calibration loaded to integrate it.")
                    else:
                        fig, ax = plt.subplots(figsize=(7.4, 5.2))
                        ax.plot(x_, y_, color="#2c3e50", lw=0.8)
                        for lo, hi in extra_ranges():
                            ax.axvspan(lo, hi, color="#c0392b", alpha=0.10)
                        ax.axhline(0, color="#aaa", lw=0.5, ls=":")
                        ax.set_xlim(float(st.session_state.rmin), float(st.session_state.rmax))
                        ax.set_xlabel("2θ (°)"); ax.set_ylabel("Counts"); ax.grid(alpha=0.2)
                        ax.set_title("Integrated, subtracted pattern (red band = masked 2θ)", fontsize=10)
                        plt.tight_layout(); _pyplot(fig)
                with st.expander("Most likely spot: noise map and zoom (opens by itself for unsure frames)",
                                 expanded=(lab_ml in ("unsure", "error"))):
                    zc1, zc2, zc3 = st.columns(3)
                    z_h = zc1.slider("Zoom half-width (pixels)", 50, 350, 100, 25, key="zoom_h3")
                    z_dy = zc2.slider("Move zoom up/down from the spot (pixels)", -300, 300, 0, 25, key="zoom_dy3")
                    z_dx = zc3.slider("Move zoom left/right from the spot (pixels)", -300, 300, 0, 25, key="zoom_dx3")
                    _CY, _CX, _H = 1437, 1440, 700
                    if min(_CY, _CX, fr_.shape[0] - _CY, fr_.shape[1] - _CX) < _H:      # smaller detector image: crop the middle
                        _CY, _CX = fr_.shape[0] // 2, fr_.shape[1] // 2
                        _H = min(700, _CY, _CX)
                    z_h = int(min(z_h, _H - 1))
                    A_ = fr_[_CY - _H:_CY + _H, _CX - _H:_CX + _H]
                    v_ = A_ != 0; w_ = v_.astype(np.float32); n_ = ndimage.uniform_filter(w_, 41) + 1e-6
                    mu_ = ndimage.uniform_filter(A_ * w_, 41) / n_
                    sd_ = np.sqrt(np.maximum(ndimage.uniform_filter(A_ * A_ * w_, 41) / n_ - mu_ * mu_, 1e-6))
                    zz_ = (A_ - mu_) / sd_; zz_[~v_] = 0
                    yy_, xx_ = np.mgrid[-_H:_H, -_H:_H]
                    z2_ = np.where(np.hypot(yy_, xx_) >= 60, zz_, 0)
                    sm_ = ndimage.uniform_filter(z2_, 3)
                    ys_, xs_ = np.unravel_index(int(np.argmax(sm_)), sm_.shape)
                    cyz_ = int(np.clip(ys_ + z_dy, z_h, 2 * _H - z_h)); cxz_ = int(np.clip(xs_ + z_dx, z_h, 2 * _H - z_h))
                    sl_ = (slice(cyz_ - z_h, cyz_ + z_h), slice(cxz_ - z_h, cxz_ + z_h))
                    fz, axz = plt.subplots(1, 3, figsize=(16, 5.4))
                    axz[0].imshow(A_, cmap="gray_r", vmin=-5, vmax=np.percentile(A_[v_], 99.5) if v_.any() else 1)
                    axz[0].plot([xs_], [ys_], "o", mfc="none", mec="red", ms=26); axz[0].set_title("Corrected frame, likely spot circled", fontsize=11)
                    axz[1].imshow(np.clip(z2_, -3, 12), cmap="magma"); axz[1].plot([xs_], [ys_], "o", mfc="none", mec="cyan", ms=26)
                    axz[1].set_title("Noise map (z)", fontsize=11)
                    cc_ = A_[sl_]; lim_ = max(np.percentile(np.abs(cc_), 99.8), 1)
                    axz[2].imshow(cc_, cmap="gray_r", vmin=-lim_ * .3, vmax=lim_); axz[2].set_title("Zoom on the spot", fontsize=11)
                    for a_ in axz:
                        a_.set_xticks([]); a_.set_yticks([])
                    plt.tight_layout(); _pyplot(fz)
                    st.caption(f"Most likely spot at pixel (x {xs_}, y {ys_}) of the crop; strongest smoothed noise value {sm_[ys_, xs_]:.1f}. "
                               "Faint hits often show as one or two such spots. Use the sliders to look around it.")
            except Exception as e:
                st.error(f"Could not show this frame: {e}")
        gj1, gj2 = st.columns([1, 4])
        goto = gj1.number_input("Go to position", 1, max(len(order), 1), int(i_ + 1), key=f"sort_goto_{cur}")
        if int(goto) != i_ + 1:
            st.session_state["sort_cur"] = order[int(goto) - 1]
            st.rerun()

    st.markdown("##### Save the hit list")
    st.caption("Writes the list of frames currently counted as hits (your label if you made one, otherwise the classifier's) "
               "and a short manifest. Tabs 3 and the live viewer read the labels directly, so this is for your records.")
    sdir = stage_dir("01b_sorting")
    if st.button("Save hit list and manifest", disabled=(sdir is None or not names), key="sort_save_hits"):
        try:
            eff_, dec_, df_ = effective_labels()
            hits_ = [n for n, l in eff_.items() if l == "hit"]
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "hit_list.txt").write_text("\n".join(hits_), encoding="utf-8")
            wm = {}
            try:
                wm = json.loads((run_dir() / "worker_manifest.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            fp.write_manifest(sdir / "sorting_manifest.json", app=APP_VERSION, method="ML classifier plus manual review of unsure frames",
                              created=datetime.datetime.now(datetime.timezone.utc).isoformat(), frames_folder=watch_dir_,
                              counts={k: sum(1 for l in eff_.values() if l == k) for k in ("hit", "non_hit", "unsure", "bad", "error")},
                              n_changed_by_user=sum(1 for n, l in dec_.items() if n in eff_ and l != df_.set_index("file").loc[n, "ml_label"]),
                              model=wm.get("model"), model_version=wm.get("model_version"), lower=wm.get("lower"), upper=wm.get("upper"))
            st.success(f"Saved {len(hits_)} hit frames to {sdir / 'hit_list.txt'}")
        except Exception as e:
            st.error(f"Could not save: {e}")
    if dec:
        with st.expander("My decisions so far"):
            _df(pd.DataFrame([(n, SORT_LABELS.get(dec[n], dec[n]), SORT_LABELS.get(ml[n][1], ml[n][1]) if n in ml else "") for n in dec],
                             columns=["file", "my label", "ML label"]), height=250)



# ===========================================================================
# TAB 3: INTEGRATE + BASELINE   (the worker has already integrated every frame; this baselines and merges)
# ===========================================================================
def subset_files(subset, lo_n=0, hi_n=0):
    """Frame names for a subset, from the effective labels (your label, else the ML label). Bad frames are always left out."""
    eff_, dec_, df_ = effective_labels()
    allow = ({"non_hit"} if subset.startswith("Non-hit") else {"hit"} if subset.startswith("Hit frames") else
             {"hit", "unsure"} if subset.startswith("Hit +") else None)
    out_ = []
    for r in df_.itertuples():
        n = str(r.file)
        lab_ = eff_[n]
        if lab_ in ("bad", "error"):
            continue
        if allow is not None and lab_ not in allow:
            continue
        if lo_n and r.collection < lo_n:
            continue
        if hi_n and r.collection > hi_n:
            continue
        out_.append(n)
    return out_, eff_, df_


with tab_int:
    st.markdown("The worker integrates every frame as it arrives (masked window bridged by a straight line) and stores the 1D pattern in "
                "`02_integrated`. This tab applies the **MOR baseline** to those stored patterns and merges them into one pattern "
                "for the run, so nothing is integrated twice. The baseline settings are in the sidebar.")
    int_subset = st.radio("Which frames", ["All frames", "Hit frames only", "Hit + unsure frames", "Non-hit frames only (control)"],
                          key="int_subset", horizontal=True,
                          help="Uses your labels where you made one, otherwise the classifier's. Bad frames and frames with an error are always left out.")
    _sc_, _ev_, _segs_ = current_segments()

    def _pick_segment():
        g = _segs_[st.session_state.int_seg]
        st.session_state["int_lo"] = int(g["start"])
        st.session_state["int_hi"] = int(g["end"]) if g["end"] is not None else 0

    if len(_segs_) > 1:
        st.selectbox("Segment (from your run events)", list(range(len(_segs_))), index=None, key="int_seg", placeholder="Pick one to fill the range below",
                     format_func=lambda i: lp.seg_label(_segs_[i]), on_change=_pick_segment)
    r1, r2 = st.columns(2)
    int_lo = r1.number_input("From collection number (0 = no limit)", 0, 10_000_000_000, 0, key="int_lo")
    int_hi = r2.number_input("To collection number (0 = no limit)", 0, 10_000_000_000, 0, key="int_hi")
    sel_names, eff_now, df_now = subset_files(int_subset, int(int_lo), int(int_hi))
    have_xy = [n for n in sel_names if xy_for(n) is not None]
    st.info(f"{len(sel_names)} frames selected, {len(have_xy)} with a stored pattern"
            + (f" ({len(sel_names) - len(have_xy)} still waiting for the worker)." if len(sel_names) > len(have_xy) else "."))
    try:
        _ranges_now = extra_ranges()
    except ValueError:
        _ranges_now = []
    st.caption(f"Masked 2θ {fp.fmt_ranges(_ranges_now) if _ranges_now else '(extra mask off)'} · MOR half_window {int(st.session_state.half_window)} · "
               f"zero the masked window after baselining: {'yes' if st.session_state.zero_masked else 'no'}")

    def baseline_one(x, y_f):
        y_b, base = fp.baseline_mor(x, y_f, int(st.session_state.half_window))
        if st.session_state.get("zero_masked", True):
            y_b = fp.zero_ranges(x, y_b, _ranges_now)
        return y_b, base

    st.markdown("##### Preview one frame")
    pv1, pv2 = st.columns([2, 1])
    prev_i = pv1.number_input("Frame number in the selection (1 = first)", 1, max(len(have_xy), 1), 1, key="int_prev_idx")
    if pv2.button("Preview baseline", disabled=not have_xy, key="int_prev_btn"):
        try:
            nm = have_xy[int(prev_i) - 1]
            x, y_f = fp.load_xy(str(xy_for(nm)))
            y_b, base = baseline_one(x, y_f)
            fig, ax = plt.subplots(1, 2, figsize=(13, 4))
            for a in ax:
                for lo, hi in _ranges_now:
                    a.axvspan(lo, hi, color="#c0392b", alpha=0.10)
                a.set_xlim(float(st.session_state.rmin), float(st.session_state.rmax)); a.grid(alpha=0.2); a.set_xlabel("2θ (°)")
            ax[0].plot(x, y_f, color="#2c3e50", lw=0.8, label="Integrated, gap filled"); ax[0].plot(x, base, "--", color="#e74c3c", lw=1.2, label="MOR baseline")
            ax[0].set_title("Stored pattern + baseline fit (red band = masked 2θ)"); ax[0].legend(fontsize=9)
            ax[1].plot(x, y_b, color="#1a7a4a", lw=1.0); ax[1].axhline(0, color="#aaa", lw=0.5, ls=":")
            ax[1].set_title(f"Baselined: {nm}"); plt.tight_layout(); _pyplot(fig)
            st.caption(f"Far-angle noise {fp.far_noise(x, y_f):.3f} counts · max after baseline {y_b.max():.1f}")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    st.markdown("##### Baseline and merge the selection")
    try:
        d03, d04 = need_output("03_baselined"), need_output("04_merged")
        st.caption(f"Writes `03_baselined` and `04_merged` under {Path(d03).parent}")
        outputs_ok = True
    except ValueError as e:
        d03 = d04 = ""
        outputs_ok = False
        st.info(str(e))
    do_merge = st.checkbox("Merge the baselined frames into one pattern for the run", value=True, key="do_merge")
    if st.button("Baseline and merge", type="primary", disabled=not (have_xy and outputs_ok), key="int_run_btn"):
        try:
            for d in (d03, d04):
                Path(d).mkdir(parents=True, exist_ok=True)
            bar = st.progress(0.0, text="Starting...")
            t0 = time.time()
            xs, ybs, rows, errors = [], [], [], []
            for n, nm in enumerate(have_xy):
                try:
                    x, y_f = fp.load_xy(str(xy_for(nm)))
                    y_b, _ = baseline_one(x, y_f)
                    fp.save_xy(Path(d03) / f"{Path(nm).stem}_baselined.xy", x, y_b, "2theta(deg) Intensity_baselined")
                    xs.append(x); ybs.append(y_b)
                    rows.append(dict(file=nm, status="OK", label=eff_now[nm], max_baselined=round(float(y_b.max()), 1),
                                     far_noise=round(fp.far_noise(x, y_f), 4)))
                except Exception as e:
                    errors.append(dict(file=nm, error=f"{type(e).__name__}: {e}"))
                bar.progress((n + 1) / len(have_xy), text=f"{n + 1}/{len(have_xy)}  {nm}")
            merged_info = None
            if do_merge and ybs:
                grid, mean_y, std_y = fp.merge_patterns(xs, ybs)
                run = clean_path(st.session_state.run_name) or "run"
                hw = int(st.session_state.half_window)
                stem = f"{run}_hw{hw}_merged_final" + ("" if int_subset.startswith("All") else "_nonhits" if int_subset.startswith("Non-hit")
                                                  else "_hits" if int_subset.startswith("Hit frames") else "_hits_unsure")
                fp.save_xy(Path(d04) / f"{stem}.xy", grid, mean_y, "2theta(deg) Intensity(averaged)")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(grid, mean_y, color="#1a7a4a", lw=1.0, label=f"Merged ({len(ybs)} frames)")
                ax.fill_between(grid, mean_y - std_y, mean_y + std_y, color="#1a7a4a", alpha=0.15, label="±1σ")
                for lo, hi in _ranges_now:
                    ax.axvspan(lo, hi, color="#c0392b", alpha=0.10)
                ax.set_xlabel("2θ (°)"); ax.set_ylabel("Intensity (a.u.)"); ax.set_xlim(float(st.session_state.rmin), float(st.session_state.rmax))
                ax.set_title(f"{run} · merged PXRD pattern ({len(ybs)} frames, red band = masked 2θ)")
                ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
                fig.savefig(Path(d04) / f"{stem}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                merged_info = dict(grid=grid, mean=mean_y, std=std_y, stem=stem)
            fp.write_manifest(Path(d03).parent / "integration_manifest.json", app=APP_VERSION,
                              created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              frames_folder=clean_path(st.session_state.watch_dir), subset=int_subset, collection_from=int(int_lo),
                              collection_to=int(int_hi), n_frames=len(rows), n_errors=len(errors),
                              extra_mask_2theta_deg=_ranges_now, baseline="MOR (pybaselines)", half_window=int(st.session_state.half_window),
                              masked_window_zeroed_after_baseline=bool(st.session_state.zero_masked),
                              gap_fill="straight line across the masked range (done by the worker)", merged=bool(merged_info),
                              labels="your label if you made one, otherwise the ML label", elapsed_s=round(time.time() - t0, 1))
            bar.progress(1.0, text=f"Done: {len(rows)} frames in {time.time() - t0:.0f} s")
            st.session_state["int_result"] = dict(rows=rows, errors=errors, merged=merged_info, d04=d04)
        except Exception as e:
            st.error(f"Run failed: {e}")

    ir = st.session_state.get("int_result")
    if ir:
        st.success(f"Baselined {len(ir['rows'])} frames." + (f" Merged pattern saved to {ir['d04']}. Open tab 4 to view it." if ir["merged"] else ""))
        if ir["errors"]:
            st.error(f"{len(ir['errors'])} file(s) failed and were skipped")
            _df(pd.DataFrame(ir["errors"]))
        if ir["merged"]:
            mg = ir["merged"]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(mg["grid"], mg["mean"], color="#1a7a4a", lw=1.0)
            ax.fill_between(mg["grid"], mg["mean"] - mg["std"], mg["mean"] + mg["std"], color="#1a7a4a", alpha=0.15)
            ax.set_xlabel("2θ (°)"); ax.set_ylabel("Intensity (a.u.)"); ax.grid(alpha=0.2); ax.set_title(mg["stem"])
            plt.tight_layout(); _pyplot(fig)
        if ir["rows"]:
            _df(pd.DataFrame(ir["rows"]), height=250)


# ===========================================================================
# TAB 4: PATTERN STACKER   (the 1D app's Pattern viewer, kept as it is)
# ===========================================================================
with tab_stack:
    st.markdown("### Pattern stacker · interactive comparison")
    st.caption("Zoom, pan and hover. Click legend entries to toggle traces. Select patterns to overlay or stack.")
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None
        st.error("plotly is not installed. Run `pip install plotly` in this app's environment.")

    if go is not None:
        stages = [n for n in ("02_integrated", "03_baselined", "04_merged")
                  if stage_dir(n) is not None and stage_dir(n).is_dir()]
        pick = st.selectbox("Stage folder in this run", ["(use the folder below)"] + stages, key="stack_stage",
                            help="Quick pick of a stage folder written by tab 3 and the run-events saves.")
        xy_dir_typed = path_input("Folder containing .xy files", "stack_dir", placeholder=r"E:/AP39/2D_output/Run7/04_merged",
                                  help="Any folder of .xy files: other runs, the 1D app's output, reference patterns.")
        xy_dir = str(stage_dir(pick)) if pick != "(use the folder below)" else xy_dir_typed
        xy_files = sorted(glob.glob(os.path.join(xy_dir, "*.xy"))) if xy_dir and os.path.isdir(xy_dir) else []
        if xy_dir and not xy_files:
            st.warning("No .xy files found in that folder.")
        if xy_files:
            selected_xy = st.multiselect("Select patterns to display", options=xy_files, default=xy_files[-1:], format_func=os.path.basename)
            co1, co2, co3, co4 = st.columns(4)
            normalise = co1.checkbox("Normalise to max = 1", value=False, help="Scale each pattern independently for shape comparison.")
            offset_mode = co2.checkbox("Stack with offset", value=False, help="Add a vertical offset between patterns for clarity.")
            offset_val = co3.number_input("Offset amount", value=1.0, step=0.1, disabled=not offset_mode)
            scale_weak = co4.number_input("Boost weak patterns (×)", value=1.0, step=0.5, min_value=0.1,
                                          help="Multiply ALL patterns by this factor before plotting.")
            with st.expander("🎨 Plot style", expanded=False):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    font_size = st.slider("Font size", 8, 24, 13, 1)
                    line_width = st.slider("Line width", 1, 5, 1, 1)
                with sc2:
                    show_grid = st.checkbox("Show gridlines", value=True)
                    show_axis_lines = st.checkbox("Show axis lines", value=True)
                    show_zero_line = st.checkbox("Show zero line (y)", value=False)
                with sc3:
                    bg_color = st.selectbox("Background", ["white", "transparent (app default)", "#f8f9fa (light grey)"], index=0)
                    plot_height = st.slider("Plot height (px)", 300, 900, 500, 50)
                legend_pos = st.selectbox("Legend position", ["Right (outside)", "Top (outside)", "Bottom (outside)", "Top-left (inside)",
                                                              "Top-right (inside)", "Bottom-left (inside)", "Bottom-right (inside)", "Hidden"], index=0)
                bg_val = {"white": "white", "transparent (app default)": "rgba(0,0,0,0)", "#f8f9fa (light grey)": "#f8f9fa"}[bg_color]
                inside = dict(bgcolor="rgba(255,255,255,0.7)")
                legend_layout_map = {
                    "Right (outside)": (dict(orientation="v", x=1.01, y=1, xanchor="left", yanchor="top"), dict(l=60, r=180, t=40, b=50)),
                    "Top (outside)": (dict(orientation="h", x=0, y=1.15, xanchor="left", yanchor="bottom"), dict(l=60, r=20, t=90, b=50)),
                    "Bottom (outside)": (dict(orientation="h", x=0, y=-0.30, xanchor="left", yanchor="top"), dict(l=60, r=20, t=40, b=110)),
                    "Top-left (inside)": (dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top", **inside), dict(l=60, r=20, t=40, b=50)),
                    "Top-right (inside)": (dict(orientation="v", x=0.99, y=0.99, xanchor="right", yanchor="top", **inside), dict(l=60, r=20, t=40, b=50)),
                    "Bottom-left (inside)": (dict(orientation="v", x=0.01, y=0.01, xanchor="left", yanchor="bottom", **inside), dict(l=60, r=20, t=40, b=50)),
                    "Bottom-right (inside)": (dict(orientation="v", x=0.99, y=0.01, xanchor="right", yanchor="bottom", **inside), dict(l=60, r=20, t=40, b=50)),
                }
                show_legend = legend_pos != "Hidden"
                legend_dict, legend_margin = legend_layout_map.get(legend_pos, legend_layout_map["Right (outside)"])

            if selected_xy:
                fig_v = go.Figure()
                colors = ["#1a7a4a", "#2980b9", "#8e44ad", "#c0392b", "#e67e22", "#16a085", "#2c3e50", "#f39c12"]
                for i, fpath in enumerate(selected_xy):
                    try:
                        xv, yv = fp.load_xy(fpath)
                        yv = np.clip(yv, 0, None)      # as in the 1D app: noise spikes must not set the scale
                        if normalise and yv.max() > 0:
                            yv = yv / yv.max()
                        if scale_weak != 1.0:
                            yv = yv * scale_weak
                        if offset_mode:
                            yv = yv + i * offset_val
                        fig_v.add_trace(go.Scatter(x=xv, y=yv, mode="lines", name=Path(fpath).stem,
                                                   line=dict(color=colors[i % len(colors)], width=line_width),
                                                   hovertemplate="2θ: %{x:.3f}°<br>I: %{y:.1f}<extra>%{fullData.name}</extra>"))
                    except Exception as e:
                        st.warning(f"Could not read {os.path.basename(fpath)}: {e}")
                axis_common = dict(showline=show_axis_lines, showgrid=show_grid, gridcolor="#e0e0e0", linecolor="black",
                                   mirror=show_axis_lines, tickfont=dict(size=font_size, color="black"),
                                   title_font=dict(size=font_size, color="black"))
                fig_v.update_layout(
                    xaxis_title="2θ (°)", yaxis_title="Intensity (a.u.)" if not normalise else "Normalised intensity",
                    xaxis=dict(range=[float(st.session_state.rmin), float(st.session_state.rmax)], zeroline=False, **axis_common),
                    yaxis=dict(zeroline=show_zero_line, zerolinecolor="#aaaaaa", zerolinewidth=1, **axis_common),
                    showlegend=show_legend, legend=dict(font=dict(size=font_size - 1, color="black"), **legend_dict),
                    font=dict(size=font_size, color="black"), hovermode="x unified", paper_bgcolor=bg_val,
                    plot_bgcolor="white", height=plot_height, margin=legend_margin)
                st.plotly_chart(fig_v, width="stretch")
                st.caption("Tip: click a pattern name in the legend to hide/show it · double-click to isolate · scroll to zoom · "
                           "drag to pan · drag the legend box itself to fine-tune its position")

                st.divider()
                st.markdown("##### Save this view")
                default_stem = ("_".join(Path(f).stem for f in selected_xy)[:80] if len(selected_xy) <= 3
                                else f"{len(selected_xy)}_patterns_comparison")
                cs1, cs2 = st.columns([3, 1])
                save_stem = cs1.text_input("Filename (no extension)", value=default_stem, key="viewer_save_stem",
                                           help="Saved as .png and .html into the folder the patterns were loaded from.")
                cs2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if cs2.button("💾 Save pattern to folder", type="primary", use_container_width=True, key="save_view_btn"):
                    if not save_stem.strip():
                        st.error("Enter a filename first.")
                    else:
                        stem = save_stem.strip(); saved = []
                        try:
                            fig_v.write_html(os.path.join(xy_dir, f"{stem}.html"), include_plotlyjs=True); saved.append(os.path.join(xy_dir, f"{stem}.html"))
                        except Exception as e:
                            st.error(f"Could not save interactive .html: {e}")
                        try:
                            fig_v.write_image(os.path.join(xy_dir, f"{stem}.png"), scale=2); saved.append(os.path.join(xy_dir, f"{stem}.png"))
                        except Exception as e:
                            st.warning(f"Could not save .png ({e}). Static export needs `pip install kaleido`. The .html saved if listed below.")
                        if saved:
                            st.success("Saved:\n" + "\n".join(f"`{p}`" for p in saved))
