"""
fep2d_app_v4.py  -  2D FEP background subtraction, OFFLINE pipeline (Streamlit)
===============================================================================
Run:    streamlit run fep2d_app_v4.py
Needs:  fep2d_v3.py and fep2d_pipeline.py in the same folder, plus numpy, scipy, h5py, pyFAI,
        pybaselines, matplotlib, pandas, plotly, streamlit  (kaleido optional, for .png export)

Sidebar   run name, output folder (every tab writes under <output>/<run name>/), calibration
          status, the extra 2-theta mask (auto-set to 5.30-6.10 deg), integration and baseline settings
Tab 0     Calibration         load image, launch pyFAI, load and verify .poni + mask (as in the 1D app)
Tab 1     Build reference     folder of reference frames -> merged reference, with a frame pop-up
Tab 2     Subtract frames     folder of raw frames -> folder of corrected 2D frames
Tab 3     Diffraction sorting   visual hit / non-hit triage of the corrected frames (ML option reserved)
Tab 4     Integrate + baseline  integrate with the mask, fill across it, MOR baseline per frame,
                              merge into one pattern for the run (as in the 1D app's Baselining tab)
Tab 5     Pattern stacker     the 1D app's pattern viewer, unchanged

The subtraction maths is fep2d_v3.py (the tested V2 maths, unchanged). Everything after the
subtraction is in fep2d_pipeline.py, which the inline processor (fep2d_inline.py) shares.

Output layout, under <output folder>/<run name>/:
    00_reference/   merged_reference.npz
    01_corrected_2D/  corrected frames, subtraction_diagnostics.csv, subtraction_manifest.json
    01b_sorting/    sorting_decisions.csv (saved on every click), scan_scores.csv, hit / non_hit / unsure copies
    02_integrated/  one .xy per frame (subtracted, gap filled across the mask, not baselined)
    03_baselined/   one .xy per frame, MOR baselined
    04_merged/      <run>_hw<N>_merged_final.xy / .png      integration_manifest.json is in the run folder
"""
import datetime
import glob
import json
import os
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

APP_VERSION = "fep2d_app_v4 / fep2d_pipeline v4 / fep2d_v3 (V2 maths unchanged)"
DEFAULT_MASK_TEXT = fp.fmt_ranges(fp.DEFAULT_EXTRA_MASK)        # "5.30-6.10"

st.set_page_config(page_title="2D FEP subtraction (offline)", page_icon="🧪", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 20px; border-radius: 6px; }
    .status-ok  { color: #1a7a4a; font-weight: 600; }
    .status-warn{ color: #e67e22; font-weight: 600; }
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


@st.cache_resource(show_spinner="Building the ring map (once per geometry and mask)...")
def _ring_map_cached(poni, poni_mt, mask_p, mask_mt, ranges_key, _ai, _mask_total):
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
# session state
# ---------------------------------------------------------------------------
for k, v in {"poni_path": "", "mask_path": "", "calib_hdf_path": "", "calib_loaded": False,
             "ai": None, "mask": None, "run_name": "", "output_dir": "",
             "use_extra_mask": True, "zero_masked": True, "extra_mask_text": DEFAULT_MASK_TEXT,
             "npt": 2000, "rmin": 1.0, "rmax": 30.0, "half_window": 4}.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Run setup")
    st.text_input("Run name", key="run_name", placeholder="Run7_CBZ_seed",
                  help="Names the folder every stage is written into, and the merged pattern.")
    path_input("Output folder", "output_dir", placeholder=r"E:/AP39/2D_output",
               help="Every tab writes into <output folder>/<run name>/. Blank folders in the tabs "
                    "below mean 'use that default'.")
    _out_base = clean_path(st.session_state.output_dir)
    if _out_base:
        st.caption(f"Outputs go to `{Path(_out_base) / (clean_path(st.session_state.run_name) or 'run')}`")
    else:
        st.markdown('<p class="status-warn">● No output folder set</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Calibration")
    if st.session_state.calib_loaded:
        st.markdown('<p class="status-ok">● Calibration ready</p>', unsafe_allow_html=True)
        st.caption(f"PONI: `{Path(st.session_state.poni_path).name}`")
        if st.session_state.mask_path:
            st.caption(f"Mask: `{Path(st.session_state.mask_path).name}`")
    else:
        st.markdown('<p class="status-warn">● No calibration loaded</p>', unsafe_allow_html=True)
        st.caption("Go to tab 0 to load and verify it.")

    st.divider()
    st.markdown("### Extra 2θ mask")
    st.checkbox("Apply extra 2θ mask (FEP reflection)", key="use_extra_mask",
                help="Masked pixels are left out of the scale fit, set to 0 in corrected frames, "
                     "ignored on integration, and bridged by a straight line in the 1D pattern.")
    st.text_input("Range(s), degrees", key="extra_mask_text", help="Several ranges: 5.30-6.10, 8.0-8.4")
    if clean_path(st.session_state.extra_mask_text) != DEFAULT_MASK_TEXT:
        if st.button(f"Reset to {DEFAULT_MASK_TEXT}", key="reset_mask"):
            st.session_state.extra_mask_text = DEFAULT_MASK_TEXT
            st.rerun()
    st.caption(f"Default {DEFAULT_MASK_TEXT}° (set 07/10/26). It removes the 5.40° spike and the "
               "leftover at 6.10°. The carbamazepine 6.10° peak sits on its edge and is lost, and "
               "any peak of another crystal form inside this window is hidden too.")
    try:
        _rng = extra_ranges()
    except ValueError as e:
        _rng = []
        st.error(f"Mask range not understood: {e}")

    with st.expander("Integration and baseline settings"):
        st.number_input("Integration points", 200, 10000, key="npt", step=100)
        c1, c2 = st.columns(2)
        c1.number_input("Min 2θ (°)", 0.0, 60.0, key="rmin", step=0.5)
        c2.number_input("Max 2θ (°)", 1.0, 80.0, key="rmax", step=0.5)
        st.checkbox("Set the masked window to 0 after baselining", key="zero_masked",
                    help="The straight-line bridge across the mask can leave a bump of a few tenths of a count "
                         "at its edges after baselining. This sets the whole masked window to 0 in the "
                         "baselined patterns (03, 04). The 02 patterns keep the straight line.")
        st.slider("MOR baseline half_window", 2, 30, key="half_window",
                  help="4 was used for every AP39 test. 5 to 10 is typical for sharp peaks.")
    st.divider()
    st.caption(APP_VERSION)

st.title("2D FEP background subtraction · offline")
(tab_cal, tab_ref, tab_sub, tab_sort, tab_int, tab_stack) = st.tabs([
    "0 · Calibration", "1 · Build reference", "2 · Subtract frames", "3 · Diffraction sorting",
    "4 · Integrate + baseline", "5 · Pattern stacker"])


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
        st.caption("Tab 2 now has this reference pre-filled.")

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


# ===========================================================================
# TAB 2: SUBTRACT FRAMES
# ===========================================================================
with tab_sub:
    st.markdown("Subtract the merged reference from **every raw frame in a folder** and write a folder of "
                "corrected 2D frames (same names, float32, negatives kept, masked pixels = 0). These are "
                "the frames for sorting (tab 3), integration (tab 4) and, later, the classifier.")
    d1, d2 = st.columns([3, 1])
    with d1:
        ref_use = path_input("Merged reference (.npz)", "ref_use", mode="file",
                             placeholder=r"E:/AP39/Run_seeded/00_reference/merged_reference.npz",
                             filetypes=[("Reference", "*.npz"), ("All files", "*.*")])
    with d2:
        sub_ext_choice = st.radio("File type", [".hdf", ".nxs", "both (.nxs + .hdf)"], key="sub_ext")
    sub_folder = path_input("Folder of raw frames", "sub_folder", placeholder=r"E:/AP39/Run_seeded/raw_frames/")
    sub_recursive = st.checkbox("Include sub-folders", value=False, key="sub_recursive")
    out_folder = path_input("Output folder for corrected 2D frames (blank = <output>/<run>/01_corrected_2D)", "out_folder")

    sub_files = []
    if sub_folder:
        try:
            sub_files = fm.find_frames(sub_folder, exts_from_choice(sub_ext_choice), sub_recursive)
            st.info(f"{len(sub_files)} raw frame files found.") if sub_files else st.warning("No matching files in that folder.")
        except FileNotFoundError as e:
            st.error(str(e))

    with st.expander("Settings", expanded=True):
        a1, a2 = st.columns(2)
        with a1:
            fit_mode = st.radio("Where the per-frame scale factor is fitted",
                                ["Whole detector (recommended)", "2θ scaling region"], key="fit_mode",
                                help="Tested on AP39: eight ways of fitting the scale gave the same result, "
                                     "so the simplest (all pixels, no offset) is the default.")
            if fit_mode.startswith("2θ"):
                f1, f2 = st.columns(2)
                fit_lo = f1.number_input("Low (°)", 0.0, 60.0, 2.0, 0.1, key="fit_lo")
                fit_hi = f2.number_input("High (°)", 0.0, 60.0, 4.0, 0.1, key="fit_hi")
        with a2:
            write_xy = st.checkbox("Also write integrated .xy patterns here", value=False, key="write_xy",
                                   help="Off by default: tab 4 integrates and baselines the corrected "
                                        "frames, with the mask, and writes the .xy files.")
            skip_existing = st.checkbox("Resume: skip frames already written", value=False, key="skip_existing")
            suffix = st.text_input("Suffix added to output names (blank keeps the original names)", value="", key="suffix")
            limit_n = st.number_input("Process only the first N frames (0 = all)", 0, 10_000_000, 0, key="limit_n")
        xy_folder_in = path_input(".xy output folder (blank = <output>/<run>/02_integrated)", "xy_folder")

    def build_settings():
        ref = get_ref(ref_use)
        ai, det = calib()
        for w in fp.check_reference(ref):
            st.warning(w)
        ranges = extra_ranges()
        total = fp.total_mask(det, ai, ref.mean.shape, ranges)
        fit_region = None
        if fit_mode.startswith("2θ"):
            if fit_hi <= fit_lo:
                raise ValueError("Scaling region: high must be greater than low.")
            fit_region = fm.tth_region(ai, ref.mean.shape, (fit_lo, fit_hi))
        return dict(ref=ref, mask=det, ai=ai, mask_total=total, fit_region=fit_region, ranges=ranges)

    ready = bool(ref_use and sub_files)

    st.markdown("#### Check the settings on one frame first")
    p1, p2, p3 = st.columns([2, 1, 1])
    prev_idx = p1.number_input("Frame number in the list (1 = first)", 1, max(len(sub_files), 1), 1, key="prev_idx")
    vmax = p2.number_input("Colour scale max (counts)", 10, 100000, 400, 50, key="vmax")
    vmin_corr = p3.number_input("Corrected scale min (counts)", -5000, 0, -100, 50, key="vmin_corr")
    if st.button("Preview before / after", disabled=not ready, key="preview_btn"):
        try:
            S = build_settings()
            p = sub_files[int(prev_idx) - 1]
            _, _, fr = next(iter(fm.iter_frames([p])))
            kw = dict(fit_region=S["fit_region"]) if S["fit_region"] is not None else {}
            t0 = time.time()
            corr, info = fm.subtract_frame(fr, S["ref"], S["mask_total"], **kw)
            dt = time.time() - t0
            fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
            show_image(ax[0], fr, 0, vmax, f"Before: {Path(p).name}", "gray", S["mask_total"])
            show_image(ax[1], corr, vmin_corr, vmax, "After: corrected", "gray", S["mask_total"])
            plt.tight_layout(); _pyplot(fig)
            st.caption(f"scale {info['scale']:.3f} · fit pixels used {info['fit_pixel_frac']:.3f} · "
                       f"negative pixels {info['neg_frac']:.3f} · subtraction {dt:.2f} s")
            if S["ranges"]:
                st.caption("Masked: detector mask plus 2θ " + ", ".join(f"{a:g}-{b:g}°" for a, b in S["ranges"]))
        except Exception as e:
            st.error(f"Preview failed: {e}")

    st.markdown("#### Four-panel check of one frame")
    st.caption("Raw, background-subtracted, significance and spot view, for the frame number chosen above.")
    if st.button("Check this frame: four-panel view", disabled=not ready, key="four_btn"):
        try:
            S = build_settings()
            p = sub_files[int(prev_idx) - 1]
            _, _, fr = next(iter(fm.iter_frames([p])))
            kw = dict(fit_region=S["fit_region"]) if S["fit_region"] is not None else {}
            t0 = time.time()
            corr, info = fm.subtract_frame(fr, S["ref"], S["mask_total"], **kw)
            t_sub = time.time() - t0
            mask_p = clean_path(st.session_state.mask_path)
            rm = _ring_map_cached(clean_path(st.session_state.poni_path), os.path.getmtime(clean_path(st.session_state.poni_path)),
                                  mask_p, os.path.getmtime(mask_p) if mask_p else 0, tuple(S["ranges"]),
                                  S["ai"], S["mask_total"])
            t0 = time.time()
            hp = fm.ring_median_subtract(corr, rm)
            t_ring = time.time() - t0
            valid = ~S["mask_total"]
            sigma = np.sqrt(np.maximum(fr, 0) + info["scale"] ** 2 * S["ref"].var_of_mean + 1e-9)
            z = corr / sigma
            spot_z = hp / sigma
            spot_z = spot_z / (1.4826 * np.median(np.abs(spot_z[valid])))
            _, n_clusters = ndimage.label(spot_z > 5)
            lo_, hi_ = np.percentile(corr[valid], [1, 99])
            mt = S["mask_total"]
            fig, ax = plt.subplots(1, 4, figsize=(30, 7.5))
            ax[0].imshow(np.where(mt, np.nan, fr), origin="lower", vmin=0, vmax=np.percentile(fr[valid], 99.5)); ax[0].set_title("Raw frame")
            ax[1].imshow(np.where(mt, np.nan, corr), origin="lower", vmin=lo_, vmax=hi_, cmap="RdBu_r"); ax[1].set_title("Background-subtracted (1-99 percentile)")
            ax[2].imshow(np.where(mt, np.nan, z), origin="lower", vmin=-6, vmax=6, cmap="RdBu_r"); ax[2].set_title("Significance (sigma)")
            ax[3].imshow(block_max(np.where(mt, 0, spot_z)), origin="lower", vmin=0, vmax=12, cmap="magma",
                         extent=[0, spot_z.shape[1], 0, spot_z.shape[0]]); ax[3].set_title("Spot view: ring median removed (block max, noise units)")
            plt.tight_layout(); _pyplot(fig)
            st.caption(f"{Path(p).name} · scale {info['scale']:.3f} · {int((spot_z > 5).sum())} pixels above 5σ in "
                       f"{n_clusters} clusters in the spot view · subtraction {t_sub:.2f} s, ring-median {t_ring:.2f} s")
        except Exception as e:
            st.error(f"Four-panel view failed: {e}")
    with st.expander("How to read the four panels"):
        st.markdown("1. **Raw**: what the detector recorded.\n"
                    "2. **Background-subtracted**: the corrected frame. Ring residuals set the colour scale, so small spots are easy to miss here.\n"
                    "3. **Significance**: corrected counts divided by the expected noise.\n"
                    "4. **Spot view**: the median of each ring is removed, so isolated spots stand out. Best for spotty hits; a smooth powder ring is removed too, so judge those from the integrated pattern.")

    st.markdown("#### Run on the whole folder")
    try:
        out_dir = need_output("01_corrected_2D", out_folder)
        xy_dir = (xy_folder_in or need_output("02_integrated")) if write_xy else None
    except ValueError as e:
        out_dir, xy_dir = "", None
        st.info(str(e))
    if out_dir:
        st.caption(f"Corrected frames go to: {out_dir}" + (f"  ·  .xy to: {xy_dir}" if xy_dir else ""))
    same_dir = bool(out_dir and sub_folder and Path(out_dir).resolve() == Path(sub_folder).resolve())
    if same_dir:
        st.error("The output folder is the same as the raw folder. Choose a different folder.")
    if st.button("Subtract reference from all frames", type="primary", disabled=(not ready) or same_dir or not out_dir, key="run_btn"):
        try:
            S = build_settings()
            files = sub_files[: int(limit_n)] if limit_n else sub_files
            bar = st.progress(0.0, text="Starting...")
            t0 = time.time()

            def _prog(done, total, name):
                el = time.time() - t0
                eta = el / done * (total - done) if done else 0
                bar.progress(done / total, text=f"{done}/{total}  {name}  ({el / done:.2f} s/file, about {eta / 60:.1f} min left)")

            kw = dict(fit_region=S["fit_region"]) if S["fit_region"] is not None else {}
            rows = fm.process_run(files, S["ref"], out_dir, mask=S["mask_total"], ai=S["ai"], xy_dir=xy_dir,
                                  write_nxs=True, npt=int(st.session_state.npt), rmin=float(st.session_state.rmin),
                                  rmax=float(st.session_state.rmax), suffix=suffix,
                                  interp_ranges=S["ranges"] or None, progress=_prog,
                                  skip_existing=skip_existing, **kw)
            errors = getattr(fm.process_run, "last_errors", [])
            elapsed = time.time() - t0
            fp.write_manifest(Path(out_dir) / "subtraction_manifest.json", app=APP_VERSION,
                              created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              raw_folder=sub_folder, output_folder=out_dir, xy_folder=xy_dir,
                              n_files_requested=len(files), n_frames_written=len(rows), n_errors=len(errors),
                              reference_file=ref_use, reference_label=S["ref"].label, reference_n_frames=S["ref"].n_used,
                              mask_file=clean_path(st.session_state.mask_path), poni_file=clean_path(st.session_state.poni_path),
                              extra_mask_2theta_deg=S["ranges"],
                              scale_fit=("whole detector" if S["fit_region"] is None else [fit_lo, fit_hi]),
                              suffix=suffix, skip_existing=skip_existing, elapsed_s=round(elapsed, 1))
            bar.progress(1.0, text=f"Done: {len(rows)} frames in {elapsed / 60:.1f} min")
            st.session_state["corrected_dir"] = out_dir
            st.session_state["batch_result"] = dict(rows=rows, errors=errors, out_dir=out_dir, elapsed=elapsed, n=len(files))
        except Exception as e:
            st.error(f"Run failed: {e}")

    br = st.session_state.get("batch_result")
    if br:
        rows, errors = br["rows"], br["errors"]
        st.success(f"Wrote {len(rows)} corrected frames to {br['out_dir']}. Tab 3 can now sort them and tab 4 integrate them.")
        if errors:
            st.error(f"{len(errors)} file(s) failed and were skipped (listed in subtraction_errors.csv)")
            _df(pd.DataFrame(errors))
        if rows:
            df = pd.DataFrame(rows)
            for c in ("scale", "neg_frac", "fit_pixel_frac"):
                df[c] = pd.to_numeric(df[c])
            med = df["scale"].median()
            mad = 1.4826 * (df["scale"] - med).abs().median() + 1e-9
            df["scale_flag"] = (df["scale"] - med).abs() > 5 * mad
            m1, m2, m3 = st.columns(3)
            m1.metric("Median scale factor", f"{med:.3f}")
            m2.metric("Frames with unusual scale", int(df["scale_flag"].sum()))
            m3.metric("Seconds per frame (incl. writing)", f"{br['elapsed'] / max(len(rows), 1):.2f}")
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(df["order"], df["scale"], ".", ms=3)
            ax.plot(df.loc[df["scale_flag"], "order"], df.loc[df["scale_flag"], "scale"], "ro", ms=5)
            ax.set_xlabel("file order"); ax.set_ylabel("scale factor"); ax.grid(alpha=0.3)
            ax.set_title("Per-frame scale factor (red = more than 5 robust σ from the median)")
            plt.tight_layout(); _pyplot(fig)
            st.caption("A scale that drifts or jumps means the intensity changed during the run or the reference no "
                       "longer matches (new solvent, detector moved). Flagged frames are worth a look.")
            _df(df.drop(columns=["order"]), height=250)


# ===========================================================================
# TAB 3: DIFFRACTION SORTING   (visual triage now; ML classifier plugs in here later)
# ===========================================================================
SORT_LABELS = {"hit": "Hit", "non_hit": "Non-hit", "unsure": "Unsure", "bad": "Bad frame"}


def sort_csv_path():
    d = stage_dir("01b_sorting")
    return None if d is None else d / "sorting_decisions.csv"


def read_decisions_csv(path):
    """{file name: label} from a decisions csv, or {} if it does not exist or cannot be read."""
    try:
        if path is not None and Path(path).is_file():
            df_ = pd.read_csv(path)
            return {str(r.file): str(r.label) for r in df_.itertuples() if str(r.label) in SORT_LABELS}
    except Exception:
        pass
    return {}


def save_decisions(dec, folder=""):
    """Write the decisions csv (called after every click so nothing is lost). False if no output folder."""
    p = sort_csv_path()
    if p is None:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(file=k, label=v, folder=folder, saved=datetime.datetime.now().isoformat(timespec="seconds"))
            for k, v in sorted(dec.items())]
    pd.DataFrame(rows, columns=["file", "label", "folder", "saved"]).to_csv(p, index=False)
    return True


def current_decisions():
    """Decisions for this run: what is on screen, else what the csv holds."""
    d = st.session_state.get("sort_dec")
    return d if d else read_decisions_csv(sort_csv_path())


def hit_score(x, y):
    """Rough guide only, NOT a classifier: tallest narrow feature above a wide running median,
    in units of the far-angle noise. FEP ripple can give non-hits a non-zero score, so the
    threshold has to be judged on your own data."""
    from scipy.ndimage import median_filter
    res = np.asarray(y, float) - median_filter(np.asarray(y, float), size=101, mode="nearest")
    noise = fp.far_noise(x, res)
    if not np.isfinite(noise) or noise <= 0:
        return float("nan")
    return float(np.nanmax(res) / noise)


@st.cache_data(show_spinner=False, max_entries=3)
def load_frame_cached(path, mtime):
    return next(iter(fm.iter_frames([path])))[2]


@st.cache_data(show_spinner=False, max_entries=1000)
def frame_pattern_cached(path, mtime, ranges_key, npt, rmin, rmax, _ai, _det):
    fr = load_frame_cached(path, mtime)
    rng = list(ranges_key)
    tmask = fp.total_mask(_det, _ai, fr.shape, rng)
    x, y = fp.integrate(fr, _ai, tmask, npt, rmin, rmax)
    y = fp.fill_gaps(x, y, rng)
    return x, y, hit_score(x, y)


def ml_classify(frame, x, y):
    """PLACEHOLDER for the trained classifier. When it exists it takes the corrected 2D frame and
    its 1D pattern and returns (label, confidence) with label in {'hit', 'non_hit', 'unsure'}.
    Its answers should fill the same decisions table as a click, so a person can still review them."""
    raise NotImplementedError("The ML classifier has not been developed yet.")


with tab_sort:
    st.markdown("Sort the corrected frames into **hit** (crystals present), **non-hit** and **unsure** by eye, "
                "before integrating. Every click is saved straight away, so you can stop and carry on later. "
                "Frames marked hit can then be the only ones integrated in tab 4. Nothing is moved or deleted.")
    so1, so2 = st.columns([3, 1])
    with so1:
        sort_folder_in = path_input("Folder of corrected 2D frames to sort (blank = the corrected frames from tab 2)",
                                    "sort_folder", placeholder=r"E:/AP39/2D_output/Run7/01_corrected_2D")
    with so2:
        sort_ext_choice = st.radio("File type", [".hdf", ".nxs", "both (.nxs + .hdf)"], key="sort_ext")
    sort_dir = (sort_folder_in or st.session_state.get("corrected_dir")
                or (str(stage_dir("01_corrected_2D")) if stage_dir("01_corrected_2D") else ""))
    sort_mode = st.radio("Sorting method", ["Visual triage (manual)", "ML classifier (not yet available)"],
                         key="sort_mode", horizontal=True,
                         help="The ML option is a placeholder. When the classifier exists it will suggest a label "
                              "for every frame into the same table, and you will still be able to review it.")
    if sort_mode.startswith("ML"):
        st.info("The ML classifier has not been developed yet, so this option does nothing for now. "
                "This is where it will plug in: it will suggest hit / non-hit for each frame into the same "
                "decisions table used below, so the rest of the pipeline will not change. "
                "Use visual triage in the meantime.")

    sort_files = []
    if sort_dir and os.path.isdir(sort_dir):
        sort_files = fm.find_frames(sort_dir, exts_from_choice(sort_ext_choice), False)
        st.info(f"{len(sort_files)} frame files found in `{sort_dir}`." if sort_files else
                f"No matching files in `{sort_dir}`. Try the other file type.")
    elif sort_dir:
        st.info(f"Folder not found yet: {sort_dir}. Run tab 2 first, or type the folder of corrected frames.")

    # decisions: load from the csv the first time for this folder / run, then keep on screen
    _key = (sort_dir, str(sort_csv_path()))
    if st.session_state.get("_sort_loaded_for") != _key:
        st.session_state["sort_dec"] = read_decisions_csv(sort_csv_path())
        st.session_state["sort_scores"] = {}
        st.session_state["sort_i"] = 0
        st.session_state["_sort_loaded_for"] = _key
        if st.session_state["sort_dec"]:
            st.toast(f"Loaded {len(st.session_state['sort_dec'])} saved decisions")
    dec = st.session_state["sort_dec"]
    scores = st.session_state["sort_scores"]
    names = [Path(p).name for p in sort_files]
    path_of = dict(zip(names, sort_files))

    if sort_csv_path() is None:
        st.warning("No output folder is set in the sidebar, so decisions are kept on screen only and will be lost "
                   "if the app restarts.")
    else:
        st.caption(f"Decisions are saved to `{sort_csv_path()}`")

    counts = {k: sum(1 for n in names if dec.get(n) == k) for k in SORT_LABELS}
    n_unl = sum(1 for n in names if n not in dec)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Frames", len(names)); m2.metric("Hit", counts["hit"]); m3.metric("Non-hit", counts["non_hit"])
    m4.metric("Unsure", counts["unsure"]); m5.metric("Bad frame", counts["bad"]); m6.metric("Not yet sorted", n_unl)
    if names:
        st.progress((len(names) - n_unl) / len(names), text=f"{len(names) - n_unl} of {len(names)} sorted")

    with st.expander("Quick screen: remove shutter-closed or failed frames in one go", expanded=False):
        st.caption("Uses the scale factors tab 2 already saved (subtraction_diagnostics.csv in the corrected-frames folder), "
                   "so it is instant. A closed-shutter or dark frame has almost no signal, so its fitted scale is far "
                   "below the rest. Flagged frames can be marked 'Bad frame' in one click and are then left out of "
                   "integration. Nothing is deleted.")
        q1, q2 = st.columns(2)
        q_frac = q1.number_input("Flag scale below this fraction of the median", 0.0, 1.0, 0.30, 0.05, key="sort_q_frac")
        q_fit = q2.number_input("Flag if the fit used fewer pixels than this fraction", 0.0, 1.0, 0.50, 0.05, key="sort_q_fit")
        _diag = Path(sort_dir) / "subtraction_diagnostics.csv" if sort_dir else None
        flagged = {}
        if _diag is not None and _diag.is_file() and names:
            try:
                dg = pd.read_csv(_diag)
                dg["scale"] = pd.to_numeric(dg["scale"], errors="coerce")
                dg["fit_pixel_frac"] = pd.to_numeric(dg["fit_pixel_frac"], errors="coerce")
                smed = float(dg["scale"].median())
                for r_ in dg.itertuples():
                    stem_ = Path(str(r_.file)).stem + (f"_f{int(r_.frame):04d}" if int(r_.frame) else "")
                    nm_ = next((n for n in names if Path(n).stem.startswith(stem_)), None)
                    if nm_ is None:
                        continue
                    why = []
                    if not np.isfinite(r_.scale) or r_.scale < q_frac * smed:
                        why.append(f"scale {r_.scale:.3f} vs median {smed:.3f}")
                    if np.isfinite(r_.fit_pixel_frac) and r_.fit_pixel_frac < q_fit:
                        why.append(f"fit used {r_.fit_pixel_frac:.2f} of pixels")
                    if why:
                        flagged[nm_] = "; ".join(why)
                if flagged:
                    st.warning(f"{len(flagged)} frame(s) flagged out of {len(dg)}.")
                    _df(pd.DataFrame([(n, w, SORT_LABELS.get(dec.get(n), "not yet sorted")) for n, w in flagged.items()],
                                     columns=["file", "why", "current label"]), height=min(250, 40 + 35 * len(flagged)))
                else:
                    st.success(f"No frames flagged out of {len(dg)}.")
            except Exception as e:
                st.error(f"Could not read the diagnostics file: {e}")
        else:
            st.info("No subtraction_diagnostics.csv found in the frames folder (it is written by tab 2).")
        if st.button("Mark flagged frames as bad", disabled=not flagged, key="sort_mark_bad"):
            for n_ in flagged:
                dec[n_] = "bad"
            save_decisions(dec, sort_dir)
            st.rerun()
    st.caption("Frames marked 'Bad frame' are skipped by tab 4 whichever option is chosen there.")

    with st.expander("Optional: quick scan to give every frame a rough score", expanded=False):
        st.caption("Integrates every frame once and scores the tallest narrow feature against the noise. "
                   "It is a guide for the review order, not a classifier. Takes about a second per frame.")
        sc1, sc2 = st.columns([1, 2])
        thr = sc2.number_input("Score above which a frame looks like a possible hit", 0.0, 1000.0, 8.0, 0.5, key="sort_thr",
                               help="A starting value. Judge it on your own frames: check what your known non-hits score.")
        if sc1.button("Scan all frames", disabled=not (names and st.session_state.calib_loaded), key="sort_scan_btn"):
            try:
                ai_, det_ = calib()
                bar = st.progress(0.0, text="Scanning...")
                rk = tuple(extra_ranges())
                for n_, nm in enumerate(names):
                    _, _, sc_ = frame_pattern_cached(path_of[nm], os.path.getmtime(path_of[nm]), rk, int(st.session_state.npt),
                                                     float(st.session_state.rmin), float(st.session_state.rmax), ai_, det_)
                    scores[nm] = sc_
                    bar.progress((n_ + 1) / len(names), text=f"{n_ + 1}/{len(names)}  {nm}")
                st.success(f"Scanned {len(names)} frames.")
                sp = sort_csv_path()
                if sp is not None:
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(dict(file=list(scores), score=list(scores.values()))).to_csv(sp.parent / "scan_scores.csv", index=False)
            except Exception as e:
                st.error(f"Scan failed: {e}")
        if not st.session_state.calib_loaded:
            st.caption("Load the calibration in tab 0 first.")
        if scores:
            fig, ax = plt.subplots(figsize=(11, 2.8))
            xs_ = np.arange(len(names))
            sv = np.array([scores.get(n, np.nan) for n in names])
            colr = [{"hit": "#1a7a4a", "non_hit": "#7f8c8d", "unsure": "#e67e22", "bad": "#c0392b"}.get(dec.get(n), "#2980b9") for n in names]
            ax.scatter(xs_, sv, c=colr, s=14)
            ax.axhline(thr, color="#c0392b", ls="--", lw=1)
            ax.set_xlabel("frame order"); ax.set_ylabel("score"); ax.grid(alpha=0.25)
            ax.set_title("Rough score per frame (green hit, grey non-hit, orange unsure, red bad, blue not yet sorted)")
            plt.tight_layout(); _pyplot(fig)

    st.markdown("##### Review")
    rv1, rv2, rv3 = st.columns(3)
    only_unl = rv1.checkbox("Only frames not yet sorted", value=False, key="sort_only_unl")
    by_score = rv2.checkbox("Highest score first (needs the scan)", value=False, key="sort_by_score", disabled=not scores)
    view_kind = rv3.radio("Image", ["Corrected frame", "Spot view (ring median removed)"], key="sort_view", horizontal=True,
                          help="The spot view removes each ring's median so isolated spots stand out. Better for spotty "
                               "hits; a smooth powder ring is removed with it, so judge those from the pattern on the right.")
    order = [n for n in names if (n not in dec or not only_unl)]
    if by_score:
        order.sort(key=lambda n: -(scores.get(n, -1) if np.isfinite(scores.get(n, -1)) else -1))
    n_ord = len(order)

    def _clamp():
        st.session_state["sort_i"] = max(0, min(int(st.session_state.get("sort_i", 0)), max(n_ord - 1, 0)))

    def _label(name, lbl):
        dec[name] = lbl
        save_decisions(dec, sort_dir)
        if not st.session_state.get("sort_only_unl"):      # the list shrinks itself when filtered
            st.session_state["sort_i"] = int(st.session_state.get("sort_i", 0)) + 1

    def _clear(name):
        dec.pop(name, None)
        save_decisions(dec, sort_dir)

    def _step(d):
        st.session_state["sort_i"] = int(st.session_state.get("sort_i", 0)) + d

    _clamp()
    if not order:
        if names:
            st.success("Every frame has been sorted. Copy them into folders below, or carry on to tab 4.")
    else:
        i_ = st.session_state["sort_i"]
        cur = order[i_]
        lab = dec.get(cur)
        tag = {"hit": "🟢 Hit", "non_hit": "⚪ Non-hit", "unsure": "🟠 Unsure", "bad": "🔴 Bad frame"}.get(lab, "⬜ not yet sorted")
        st.markdown(f"**{cur}**  ·  frame {i_ + 1} of {n_ord}  ·  {tag}")
        b1, b2, b3, b4, b7, b5, b6 = st.columns(7)
        b1.button("◀ Back", key="sort_back", on_click=_step, args=(-1,), disabled=i_ == 0, use_container_width=True)
        b2.button("🟢 Hit", key="sort_hit", on_click=_label, args=(cur, "hit"), type="primary", use_container_width=True)
        b3.button("⚪ Non-hit", key="sort_non", on_click=_label, args=(cur, "non_hit"), use_container_width=True)
        b4.button("🟠 Unsure", key="sort_uns", on_click=_label, args=(cur, "unsure"), use_container_width=True)
        b7.button("🔴 Bad frame", key="sort_bad", on_click=_label, args=(cur, "bad"), use_container_width=True)
        b5.button("Skip ▶", key="sort_skip", on_click=_step, args=(1,), disabled=i_ >= n_ord - 1, use_container_width=True)
        b6.button("Clear label", key="sort_clear", on_click=_clear, args=(cur,), disabled=lab is None, use_container_width=True)
        if not st.session_state.calib_loaded:
            st.warning("Load the calibration in tab 0 to see the frame and its pattern.")
        else:
            try:
                ai_, det_ = calib()
                rk = tuple(extra_ranges())
                pth = path_of[cur]
                mt_ = os.path.getmtime(pth)
                fr_ = load_frame_cached(pth, mt_)
                tm_ = fp.total_mask(det_, ai_, fr_.shape, list(rk))
                x_, y_, sc_ = frame_pattern_cached(pth, mt_, rk, int(st.session_state.npt), float(st.session_state.rmin),
                                                   float(st.session_state.rmax), ai_, det_)
                scores[cur] = sc_
                valid_ = ~tm_
                cA, cB = st.columns([1, 1.25])
                with cA:
                    if view_kind.startswith("Spot"):
                        mp_ = clean_path(st.session_state.mask_path)
                        pp_ = clean_path(st.session_state.poni_path)
                        rm_ = _ring_map_cached(pp_, os.path.getmtime(pp_), mp_, os.path.getmtime(mp_) if mp_ else 0, rk, ai_, tm_)
                        hp_ = fm.ring_median_subtract(fr_, rm_)
                        mad_ = 1.4826 * np.median(np.abs(hp_[valid_] - np.median(hp_[valid_]))) + 1e-9
                        img_, lo_, hi_, cm_ = np.where(tm_, 0, hp_ / mad_), 0, 12, "magma"
                        ttl_ = "Spot view (noise units)"
                        img_ = block_max(img_, 2)
                        tmk_ = None
                    else:
                        lo_, hi_ = np.percentile(fr_[valid_], [1, 99.5])
                        img_, cm_, ttl_, tmk_ = fr_, "gray", "Corrected frame (masked = grey)", tm_
                        img_ = np.where(tmk_, np.nan, img_)[::2, ::2]
                    fig, ax = plt.subplots(figsize=(5.2, 5.2))
                    ax.imshow(img_, origin="lower", vmin=lo_, vmax=hi_, cmap=cm_)
                    ax.set_title(ttl_, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                    plt.tight_layout(); _pyplot(fig)
                with cB:
                    fig, ax = plt.subplots(figsize=(7.4, 5.2))
                    ax.plot(x_, y_, color="#2c3e50", lw=0.8)
                    for lo, hi in extra_ranges():
                        ax.axvspan(lo, hi, color="#c0392b", alpha=0.10)
                    ax.axhline(0, color="#aaa", lw=0.5, ls=":")
                    ax.set_xlim(float(st.session_state.rmin), float(st.session_state.rmax))
                    ax.set_xlabel("2θ (°)"); ax.set_ylabel("Counts"); ax.grid(alpha=0.2)
                    ax.set_title("Integrated, subtracted pattern (red band = masked 2θ)", fontsize=10)
                    plt.tight_layout(); _pyplot(fig)
                msg = f"Rough score {sc_:.1f}"
                if np.isfinite(sc_):
                    msg += f" · {'above' if sc_ > thr else 'below'} the {thr:g} guide value" if scores else ""
                st.caption(msg + ". The score is only a guide; the decision is yours.")
            except Exception as e:
                st.error(f"Could not show this frame: {e}")
        gj1, gj2 = st.columns([1, 4])
        goto = gj1.number_input("Go to position", 1, max(n_ord, 1), int(i_ + 1), key=f"sort_goto_{i_}")
        if int(goto) != i_ + 1:
            st.session_state["sort_i"] = int(goto) - 1
            st.rerun()

    st.markdown("##### Copy the sorted frames into folders")
    st.caption("Copies (never moves or deletes) the labelled frames into hit, non_hit, unsure and bad folders "
               "and writes a list of the hit frames.")
    sd = stage_dir("01b_sorting")
    if st.button("Copy sorted frames", disabled=(sd is None or not any(n in dec for n in names)), key="sort_copy_btn"):
        try:
            import shutil
            ncopy = {k: 0 for k in SORT_LABELS}
            for nm in names:
                if dec.get(nm) in SORT_LABELS:
                    dst = sd / dec[nm]
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path_of[nm], dst / nm)
                    ncopy[dec[nm]] += 1
            (sd / "hit_list.txt").write_text("\n".join(n for n in names if dec.get(n) == "hit"))
            fp.write_manifest(sd / "sorting_manifest.json", app=APP_VERSION, method="visual triage",
                              created=datetime.datetime.now(datetime.timezone.utc).isoformat(), frames_folder=sort_dir,
                              counts=ncopy, not_sorted=sum(1 for n in names if n not in dec))
            st.success(f"Copied {ncopy['hit']} hit, {ncopy['non_hit']} non-hit, {ncopy['unsure']} unsure and {ncopy['bad']} bad frames to {sd}")
        except Exception as e:
            st.error(f"Copy failed: {e}")
    if sd is None:
        st.caption("Set the output folder in the sidebar to enable this.")
    if dec:
        with st.expander("Decisions so far"):
            _df(pd.DataFrame([(n, SORT_LABELS.get(dec[n], dec[n])) for n in names if n in dec], columns=["file", "label"]), height=250)


# ===========================================================================
# TAB 4: INTEGRATE + BASELINE
# ===========================================================================
with tab_int:
    st.markdown("Integrate the corrected 2D frames to 1D with the mask, bridge the masked 2θ range with a "
                "straight line, MOR-baseline every frame, then merge them into one pattern for the run. "
                "This follows the 1D app's Baselining tab, with the extra mask applied at integration.")
    i1, i2 = st.columns([3, 1])
    with i1:
        int_folder = path_input("Folder of 2D frames to integrate (blank = the corrected frames from tab 2)",
                                "int_folder", placeholder=r"E:/AP39/2D_output/Run7/01_corrected_2D")
    with i2:
        int_ext_choice = st.radio("File type", [".hdf", ".nxs", "both (.nxs + .hdf)"], key="int_ext",
                                  help="Corrected frames keep the extension of the raw frames.")
    int_dir_eff = int_folder or st.session_state.get("corrected_dir") or (str(stage_dir("01_corrected_2D")) if stage_dir("01_corrected_2D") else "")
    int_files = []
    if int_dir_eff and os.path.isdir(int_dir_eff):
        int_files = fm.find_frames(int_dir_eff, exts_from_choice(int_ext_choice), False)
        st.info(f"{len(int_files)} frame files found in `{int_dir_eff}`." if int_files else
                f"No matching files in `{int_dir_eff}`. Try the other file type.")
    elif int_dir_eff:
        st.info(f"Folder not found yet: {int_dir_eff}. Run tab 2 first, or type the folder of frames to integrate.")

    int_subset = st.radio("Which frames", ["All frames in the folder", "Hit frames only (from tab 3)", "Hit + unsure (from tab 3)",
                           "Non-hit frames only (control, from tab 3)"],
                          key="int_subset", horizontal=True)
    _dec0 = current_decisions()
    _nbad = sum(1 for p in int_files if _dec0.get(Path(p).name) == "bad")
    if _nbad:
        int_files = [p for p in int_files if _dec0.get(Path(p).name) != "bad"]
        st.caption(f"{_nbad} frame(s) marked 'Bad frame' in tab 3 are left out.")
    if not int_subset.startswith("All"):
        _dec = current_decisions()
        _allow = ({"non_hit"} if int_subset.startswith("Non-hit") else {"hit"} if "unsure" not in int_subset
                  else {"hit", "unsure"})
        _kept = [p for p in int_files if _dec.get(Path(p).name) in _allow]
        if _dec:
            st.info(f"{len(_kept)} of {len(int_files)} frames selected from the tab 3 sorting.")
        else:
            st.warning("No sorting decisions found for this run. Sort frames in tab 3 first, or choose 'All frames'.")
        int_files = _kept

    try:
        _ranges_now = extra_ranges()
    except ValueError:
        _ranges_now = []
    st.caption(f"Mask: detector mask plus 2θ {fp.fmt_ranges(_ranges_now) if _ranges_now else '(extra mask off)'} · "
               f"{int(st.session_state.npt)} points · {st.session_state.rmin:g}-{st.session_state.rmax:g}° · "
               f"MOR half_window {int(st.session_state.half_window)} (change in the sidebar)")

    def int_pipeline(frame, ai, tmask):
        x, y = fp.integrate(frame, ai, tmask, int(st.session_state.npt), float(st.session_state.rmin), float(st.session_state.rmax))
        y_f = fp.fill_gaps(x, y, _ranges_now)
        y_b, base = fp.baseline_mor(x, y_f, int(st.session_state.half_window))
        if st.session_state.get("zero_masked", True):
            y_b = fp.zero_ranges(x, y_b, _ranges_now)
        return x, y_f, y_b, base

    st.markdown("##### Preview one frame")
    pv1, pv2 = st.columns([2, 1])
    int_prev_idx = pv1.number_input("Frame number in the list (1 = first)", 1, max(len(int_files), 1), 1, key="int_prev_idx")
    if pv2.button("Preview baseline", disabled=not int_files, key="int_prev_btn"):
        try:
            ai, det = calib()
            p = int_files[int(int_prev_idx) - 1]
            _, _, fr = next(iter(fm.iter_frames([p])))
            tmask = fp.total_mask(det, ai, fr.shape, _ranges_now)
            x, y_f, y_b, base = int_pipeline(fr, ai, tmask)
            fig, ax = plt.subplots(1, 2, figsize=(13, 4))
            for a in ax:
                for lo, hi in _ranges_now:
                    a.axvspan(lo, hi, color="#c0392b", alpha=0.10)
                a.set_xlim(float(st.session_state.rmin), float(st.session_state.rmax)); a.grid(alpha=0.2); a.set_xlabel("2θ (°)")
            ax[0].plot(x, y_f, color="#2c3e50", lw=0.8, label="Integrated, gap filled"); ax[0].plot(x, base, "--", color="#e74c3c", lw=1.2, label="MOR baseline")
            ax[0].set_title("Integrated + baseline fit (red band = masked 2θ)"); ax[0].legend(fontsize=9)
            ax[1].plot(x, y_b, color="#1a7a4a", lw=1.0); ax[1].axhline(0, color="#aaa", lw=0.5, ls=":")
            ax[1].set_title(f"Baselined: {Path(p).name}"); plt.tight_layout(); _pyplot(fig)
            st.caption(f"Far-angle noise {fp.far_noise(x, y_f):.3f} counts · max after baseline {y_b.max():.1f}")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    st.markdown("##### Run on the whole folder")
    try:
        d02, d03, d04 = need_output("02_integrated"), need_output("03_baselined"), need_output("04_merged")
        st.caption(f"Writes `02_integrated`, `03_baselined` and `04_merged` under {Path(d02).parent}")
        outputs_ok = True
    except ValueError as e:
        d02 = d03 = d04 = ""
        outputs_ok = False
        st.info(str(e))
    do_merge = st.checkbox("Merge the baselined frames into one pattern for the run", value=True, key="do_merge")
    if st.button("Integrate, baseline and merge all frames", type="primary", disabled=not (int_files and outputs_ok), key="int_run_btn"):
        try:
            ai, det = calib()
            _, _, fr0 = next(iter(fm.iter_frames([int_files[0]])))
            tmask = fp.total_mask(det, ai, fr0.shape, _ranges_now)
            for d in (d02, d03, d04):
                Path(d).mkdir(parents=True, exist_ok=True)
            bar = st.progress(0.0, text="Starting...")
            t0 = time.time()
            xs, ybs, rows, errors = [], [], [], []
            for n, p in enumerate(int_files):
                try:
                    nfr = fm.n_frames_in_file(p)
                    for _, i, fr in fm.iter_frames([p]):
                        stem = Path(p).stem + (f"_f{i:04d}" if nfr > 1 else "")
                        x, y_f, y_b, _ = int_pipeline(fr, ai, tmask)
                        fp.save_xy(Path(d02) / f"{stem}.xy", x, y_f, "2theta(deg) Intensity_2Dcorr")
                        fp.save_xy(Path(d03) / f"{stem}_baselined.xy", x, y_b, "2theta(deg) Intensity_baselined")
                        xs.append(x); ybs.append(y_b)
                        rows.append(dict(file=stem, status="OK", max_baselined=round(float(y_b.max()), 1),
                                         far_noise=round(fp.far_noise(x, y_f), 4)))
                except Exception as e:
                    errors.append(dict(file=Path(p).name, error=f"{type(e).__name__}: {e}"))
                el = time.time() - t0
                bar.progress((n + 1) / len(int_files), text=f"{n + 1}/{len(int_files)}  {Path(p).name}  ({el / (n + 1):.2f} s/file)")
            merged_info = None
            if do_merge and ybs:
                grid, mean_y, std_y = fp.merge_patterns(xs, ybs)
                run = clean_path(st.session_state.run_name) or "run"
                hw = int(st.session_state.half_window)
                stem = f"{run}_hw{hw}_merged_final" + ("" if int_subset.startswith("All") else "_nonhits" if int_subset.startswith("Non-hit")
                                                  else "_hits" if "unsure" not in int_subset else "_hits_unsure")
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
                merged_info = dict(grid=grid, mean=mean_y, std=std_y, stem=stem)
            fp.write_manifest(Path(d02).parent / "integration_manifest.json", app=APP_VERSION,
                              created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              frames_folder=int_dir_eff, subset=int_subset, n_frames=len(rows), n_errors=len(errors),
                              extra_mask_2theta_deg=_ranges_now, npt=int(st.session_state.npt),
                              rmin=float(st.session_state.rmin), rmax=float(st.session_state.rmax),
                              baseline="MOR (pybaselines)", half_window=int(st.session_state.half_window),
                              masked_window_zeroed_after_baseline=bool(st.session_state.zero_masked),
                              gap_fill="straight line across the masked range", merged=bool(merged_info),
                              mask_file=clean_path(st.session_state.mask_path), poni_file=clean_path(st.session_state.poni_path),
                              elapsed_s=round(time.time() - t0, 1))
            bar.progress(1.0, text=f"Done: {len(rows)} frames in {(time.time() - t0) / 60:.1f} min")
            st.session_state["int_result"] = dict(rows=rows, errors=errors, merged=merged_info, d04=d04)
        except Exception as e:
            st.error(f"Run failed: {e}")

    ir = st.session_state.get("int_result")
    if ir:
        st.success(f"Integrated and baselined {len(ir['rows'])} frames."
                   + (f" Merged pattern saved to {ir['d04']}. Open tab 5 to view it." if ir["merged"] else ""))
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
                            help="Quick pick of a stage folder written by tab 4.")
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
