"""
fep2d_app_v3_compare.py  -  2D FEP background subtraction (Streamlit)
=====================================================================
Tab 1  Build reference : folder of 2D reference frames -> one merged reference (.npz)
Tab 2  Subtract frames : folder of raw frames -> folder of corrected 2D frames
                         (+ optional .xy), with a run manifest for provenance.
Tab 3  Compare with 1D : the 1D-subtracted .xy against the 2D-corrected .xy of the same
                         run(s): peak positions and heights, noise, baseline, mask edges.

Tabs 1 and 2 are fep2d_app_v3.py unchanged (that file is untouched). Tab 3 is new and
only reads .xy files; it never touches the 2D frames.

Run:   streamlit run fep2d_app_v3_compare.py
Needs: fep2d_v3.py and fep2d_compare.py in the same folder, plus numpy, scipy, h5py,
       pyFAI, matplotlib, pandas, streamlit.

The maths is the tested V2 trial (Trial_2D_Diagnose_and_Output.ipynb):
merged reference, per-frame robust scale fit, optional 2theta mask, float32
output with negatives kept. Nothing here changes that.
"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fep2d_v3 as fm  # noqa: E402
import fep2d_compare as fc  # noqa: E402

APP_VERSION = "fep2d_app_v3_compare / fep2d_v3 / fep2d_compare (V2 maths unchanged)"

st.set_page_config(page_title="2D FEP subtraction", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def parse_ranges(text: str):
    """'5.40-6.00, 8-8.5' -> [(5.40, 6.00), (8.0, 8.5)]. Raises ValueError."""
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-")
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError(f"range '{part}' must have high > low")
        out.append((lo, hi))
    return out


def clean_path(s: str) -> str:
    return s.strip().strip('"').strip("'")


def exts_from_choice(choice: str):
    return {".nxs": (".nxs",), ".hdf": (".hdf", ".h5", ".hdf5"),
            "both (.nxs + .hdf)": (".nxs", ".hdf", ".h5", ".hdf5")}[choice]


@st.cache_resource(show_spinner=False)
def _load_mask(path: str, mtime: float):
    return np.load(path).astype(bool)


@st.cache_resource(show_spinner=False)
def _load_poni(path: str, mtime: float):
    import pyFAI
    return pyFAI.load(path)


@st.cache_resource(show_spinner=False)
def _load_ref(path: str, mtime: float):
    return fm.load_reference(path)


def get_mask(path: str):
    path = clean_path(path)
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Detector mask not found: {path}")
    return _load_mask(path, os.path.getmtime(path))


def get_ai(path: str):
    path = clean_path(path)
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f".poni file not found: {path}")
    return _load_poni(path, os.path.getmtime(path))


def get_ref(path: str):
    path = clean_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference file not found: {path}")
    return _load_ref(path, os.path.getmtime(path))


@st.cache_resource(show_spinner="Building the ring map (once per geometry and mask)...")
def _ring_map_cached(poni, poni_mt, mask_p, mask_mt, ranges_key, _ai, _mask_total):
    """Arguments starting with _ are not hashed; the others form the cache key."""
    return fm.build_ring_map(_ai, _mask_total.shape, _mask_total)


def block_max(a, b=4):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).max(axis=(1, 3))


def show_image(ax, a, vmin, vmax, title, cmap="gray", mask=None):
    img = np.where(mask, np.nan, a) if mask is not None else a
    ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# sidebar: geometry shared by both tabs
# ---------------------------------------------------------------------------
st.sidebar.header("Detector and geometry")
st.sidebar.caption("Set these for the beamtime or data set you are working on. "
                   "Each beamtime needs its own mask and .poni.")
mask_path = st.sidebar.text_input("Detector mask (.npy, True = bad pixel)",
                                  key="mask_path", placeholder=r"E:/calib/mask.npy")
poni_path = st.sidebar.text_input(".poni geometry file (optional but needed for the "
                                  "2θ mask, scaling region and .xy)",
                                  key="poni_path", placeholder=r"E:/calib/calib.poni")
use_extra_mask = st.sidebar.checkbox("Apply extra 2θ mask (FEP reflection)", value=True,
                                     key="use_extra_mask",
                                     help="Needs the .poni. Masked pixels are left out of the "
                                          "scale fit, set to 0 in the corrected frame and ignored "
                                          "on integration.")
extra_mask_text = st.sidebar.text_input("Extra mask 2θ range(s), degrees", value="5.40-6.00",
                                        key="extra_mask_text",
                                        help="Several ranges: 5.40-6.00, 8.0-8.4")
st.sidebar.divider()
st.sidebar.caption(APP_VERSION)

st.title("2D FEP background subtraction")
tab_ref, tab_sub, tab_cmp = st.tabs(["1 · Build reference", "2 · Subtract frames", "3 · Compare with 1D"])


# ===========================================================================
# TAB 1: BUILD REFERENCE
# ===========================================================================
with tab_ref:
    st.markdown(
        "Merge every 2D reference frame in a folder into **one reference frame** "
        "(per-pixel mean and its variance). Use frames of the solvent in the FEP tube "
        "with no crystals, for example the 70:30 reference frames collected at the start "
        "of a run, or non-hit frames when no dedicated reference exists."
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        ref_folder = clean_path(st.text_input("Folder of 2D reference frames", key="ref_folder",
                                              placeholder=r"E:/AP39/Run_seeded/reference_frames/"))
    with c2:
        ref_ext_choice = st.radio("File type", [".nxs", ".hdf", "both (.nxs + .hdf)"],
                                  key="ref_ext", help="Use .nxs unless the folder holds only "
                                  ".hdf. Choosing both counts every frame twice when each .nxs "
                                  "links to a .hdf.")
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
    with c3:
        ref_label = st.text_input("Reference label (stored in the file and in every corrected "
                                  "frame)", value="70:30 EtOH:H2O merged", key="ref_label")
    with c4:
        ref_out = clean_path(st.text_input("Save reference as (blank = <folder>/merged_reference.npz)",
                                           key="ref_out"))

    screen_on = st.checkbox("Screen out bad frames before merging", value=True, key="screen_on",
                            help="Rejects shutter-closed frames, total-count outliers and frames "
                                 "whose pattern does not match the rest (detector out of position).")
    with st.expander("Screening thresholds"):
        s1, s2, s3 = st.columns(3)
        min_total_frac = s1.number_input("Minimum total counts, fraction of median", 0.0, 1.0,
                                         0.30, 0.05, key="min_total_frac")
        z_total = s2.number_input("Total-count outlier limit (robust σ)", 1.0, 20.0, 5.0, 0.5,
                                  key="z_total")
        min_corr = s3.number_input("Minimum pattern correlation with the median frame", 0.0, 1.0,
                                   0.97, 0.01, key="min_corr")

    if st.button("Build merged reference", type="primary", disabled=not ref_files,
                 key="build_ref_btn"):
        try:
            mask = get_mask(mask_path)
            out_path = ref_out or str(Path(ref_folder) / "merged_reference.npz")
            if not out_path.lower().endswith(".npz"):
                out_path += ".npz"
            bar = st.progress(0.0, text="Starting...")
            t0 = time.time()

            def _prog(stage, done, total):
                frac = min(1.0, done / max(total, 1))
                # screening is the first half of the work, merging the second
                bar.progress(0.5 * frac if stage == "screening" else 0.5 + 0.5 * frac,
                             text=f"{stage}: {done}/{total}")

            kw = dict(min_total_frac=min_total_frac, z_total=z_total, min_corr=min_corr) if screen_on else {}
            ref = fm.build_reference(ref_files, mask=mask, label=ref_label, screen=screen_on,
                                     progress=_prog, **kw)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fm.save_reference(ref, out_path)
            bar.progress(1.0, text=f"Done in {time.time() - t0:.0f} s")
            st.session_state["last_ref_path"] = out_path
            st.session_state["ref_use"] = out_path      # pre-fills tab 2
            st.session_state["ref_result"] = dict(
                path=out_path, n_used=ref.n_used, n_rejected=ref.n_rejected,
                rejected=ref.rejected, shape=ref.mean.shape, label=ref.label)
            st.session_state["ref_preview"] = (ref.mean.copy(), ref.var_of_mean.copy(), mask)
        except Exception as e:
            st.error(f"Could not build the reference: {e}")

    res = st.session_state.get("ref_result")
    if res:
        st.success(f"Saved merged reference to {res['path']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames merged", res["n_used"])
        m2.metric("Frames rejected", res["n_rejected"])
        m3.metric("Frame size (px)", f"{res['shape'][1]} × {res['shape'][0]}")
        if res["n_used"] < 30:
            st.warning(f"Only {res['n_used']} frames were merged. In AP39, 93 to 112 frames "
                       "gave a good reference and 5 frames was visibly too few.")
        if res["rejected"]:
            st.markdown("**Rejected frames**")
            st.dataframe(pd.DataFrame([(Path(p).name, i, why) for p, i, why in res["rejected"]],
                                      columns=["file", "frame", "reason"]), use_container_width=True)
        mean_, var_, mask_ = st.session_state["ref_preview"]
        v_ = ~mask_ if mask_ is not None else np.ones(mean_.shape, bool)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        show_image(ax[0], mean_, np.percentile(mean_[v_], 1), np.percentile(mean_[v_], 99.5),
                   f"Merged reference ({res['n_used']} frames)", "viridis", mask_)
        snr = mean_[v_] / np.sqrt(var_[v_] + 1e-12)
        ax[1].hist(snr, bins=100, log=True); ax[1].set_xlabel("per-pixel SNR (mean / std error)")
        ax[1].set_title("Reference noise")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Tab 2 now has this reference pre-filled.")


# ===========================================================================
# TAB 2: SUBTRACT FRAMES
# ===========================================================================
with tab_sub:
    st.markdown(
        "Subtract the merged reference from **every raw frame in a folder** and write a "
        "folder of corrected 2D frames (same names, same layout, float32, negatives kept, "
        "masked pixels = 0). These are the frames for the classifier."
    )
    d1, d2 = st.columns([3, 1])
    with d1:
        ref_use = clean_path(st.text_input(
            "Merged reference (.npz)", key="ref_use",
            placeholder=r"E:/AP39/Run_seeded/reference_frames/merged_reference.npz"))
    with d2:
        sub_ext_choice = st.radio("File type", [".nxs", ".hdf", "both (.nxs + .hdf)"], key="sub_ext")
    sub_folder = clean_path(st.text_input("Folder of raw frames", key="sub_folder",
                                          placeholder=r"E:/AP39/Run_seeded/raw_frames/"))
    sub_recursive = st.checkbox("Include sub-folders", value=False, key="sub_recursive")
    out_folder = clean_path(st.text_input(
        "Output folder for corrected 2D frames (blank = <raw folder>_corrected_2D)", key="out_folder"))

    sub_files = []
    if sub_folder:
        try:
            sub_files = fm.find_frames(sub_folder, exts_from_choice(sub_ext_choice), sub_recursive)
            if sub_files:
                st.info(f"{len(sub_files)} raw frame files found.")
            else:
                st.warning("No matching files in that folder.")
        except FileNotFoundError as e:
            st.error(str(e))

    with st.expander("Settings", expanded=True):
        a1, a2 = st.columns(2)
        with a1:
            fit_mode = st.radio(
                "Where the per-frame scale factor is fitted",
                ["Whole detector (recommended)", "2θ scaling region"], key="fit_mode",
                help="Tested on AP39: the 2 to 4° region gave no clear gain over the whole "
                     "detector (mean peak-to-noise 19.6 against 19.2).")
            if fit_mode.startswith("2θ"):
                f1, f2 = st.columns(2)
                fit_lo = f1.number_input("Low (°)", 0.0, 60.0, 2.0, 0.1, key="fit_lo")
                fit_hi = f2.number_input("High (°)", 0.0, 60.0, 4.0, 0.1, key="fit_hi")
        with a2:
            write_xy = st.checkbox("Also write integrated .xy patterns", value=True, key="write_xy",
                                   help="Needs the .poni. 2000 points, 1 to 30°, no solid-angle "
                                        "correction, masked 2θ range filled by a straight line.")
            skip_existing = st.checkbox("Resume: skip frames already written", value=False,
                                        key="skip_existing")
            suffix = st.text_input("Suffix added to output names (blank keeps the original names)",
                                   value="", key="suffix")
            limit_n = st.number_input("Process only the first N frames (0 = all)", 0, 10_000_000, 0,
                                      key="limit_n")
        xy_folder_in = clean_path(st.text_input(
            ".xy output folder (blank = <output folder>_xy)", key="xy_folder"))
        i1, i2, i3 = st.columns(3)
        npt = i1.number_input("Integration points", 200, 10000, 2000, 100, key="npt")
        rmin = i2.number_input("Min 2θ (°)", 0.0, 60.0, 1.0, 0.5, key="rmin")
        rmax = i3.number_input("Max 2θ (°)", 1.0, 80.0, 30.0, 0.5, key="rmax")

    # ---- build the settings shared by preview and batch ----
    def build_settings():
        """Returns dict(ref, mask, ai, mask_total, fit_region, interp_ranges) or raises."""
        ref = get_ref(ref_use)
        mask = get_mask(mask_path)
        ai = get_ai(poni_path)
        if mask is not None and mask.shape != ref.mean.shape:
            raise ValueError(f"Mask is {mask.shape} but the reference frames are {ref.mean.shape}. "
                             "Use the mask for this detector set-up.")
        ranges = []
        if use_extra_mask:
            try:
                ranges = parse_ranges(extra_mask_text)
            except ValueError as e:
                raise ValueError(f"Extra mask range not understood ({e}). Use e.g. 5.40-6.00")
            if ranges and ai is None:
                raise ValueError("The extra 2θ mask needs the .poni file. Add it in the sidebar "
                                 "or untick the extra mask.")
        total = mask.copy() if mask is not None else np.zeros(ref.mean.shape, bool)
        if ranges:
            total |= fm.tth_annulus_mask(ai, ref.mean.shape, ranges)
        fit_region = None
        if fit_mode.startswith("2θ"):
            if ai is None:
                raise ValueError("A 2θ scaling region needs the .poni file.")
            if fit_hi <= fit_lo:
                raise ValueError("Scaling region: high must be greater than low.")
            fit_region = fm.tth_region(ai, ref.mean.shape, (fit_lo, fit_hi))
        if write_xy and ai is None:
            raise ValueError("Writing .xy needs the .poni file. Add it in the sidebar or untick .xy.")
        return dict(ref=ref, mask=mask, ai=ai, mask_total=total, fit_region=fit_region,
                    ranges=ranges)

    ready = bool(ref_use and sub_files)

    # ---- preview one frame ----
    st.markdown("#### Check the settings on one frame first")
    p1, p2, p3 = st.columns([2, 1, 1])
    prev_idx = p1.number_input("Frame number in the list (1 = first)", 1, max(len(sub_files), 1), 1,
                               key="prev_idx")
    vmax = p2.number_input("Colour scale max (counts)", 10, 100000, 400, 50, key="vmax")
    vmin_corr = p3.number_input("Corrected scale min (counts)", -5000, 0, -100, 50, key="vmin_corr")
    if st.button("Preview before / after", disabled=not ready, key="preview_btn"):
        try:
            S = build_settings()
            p = sub_files[int(prev_idx) - 1]
            frames = list(fm.iter_frames([p]))
            _, fidx, fr = frames[0]
            kw = dict(fit_region=S["fit_region"]) if S["fit_region"] is not None else {}
            t0 = time.time()
            corr, info = fm.subtract_frame(fr, S["ref"], S["mask_total"], **kw)
            dt = time.time() - t0
            fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
            show_image(ax[0], fr, 0, vmax, f"Before: {Path(p).name}", "gray", S["mask_total"])
            show_image(ax[1], corr, vmin_corr, vmax, "After: corrected", "gray", S["mask_total"])
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption(f"scale {info['scale']:.3f} · fit pixels used {info['fit_pixel_frac']:.3f} · "
                       f"negative pixels {info['neg_frac']:.3f} · subtraction {dt:.2f} s")
            if S["ranges"]:
                st.caption("Masked: detector mask plus 2θ " +
                           ", ".join(f"{a:g}-{b:g}°" for a, b in S["ranges"]))
        except Exception as e:
            st.error(f"Preview failed: {e}")

    # ---- four-panel check (same as Step 6 of the notebook) ----
    st.markdown("#### Four-panel check of one frame")
    st.caption("Raw, background-subtracted, significance and spot view, for the frame number "
               "chosen above. Needs the .poni. This is the notebook's Step 6.")
    if st.button("Check this frame: four-panel view", disabled=not ready, key="four_btn"):
        try:
            S = build_settings()
            if S["ai"] is None:
                raise ValueError("The four-panel view needs the .poni file (the spot view uses "
                                 "each pixel's distance from the beam centre).")
            p = sub_files[int(prev_idx) - 1]
            _, _, fr = next(iter(fm.iter_frames([p])))
            kw = dict(fit_region=S["fit_region"]) if S["fit_region"] is not None else {}
            t0 = time.time()
            corr, info = fm.subtract_frame(fr, S["ref"], S["mask_total"], **kw)
            t_sub = time.time() - t0
            mask_p = clean_path(mask_path)
            rm = _ring_map_cached(clean_path(poni_path), os.path.getmtime(clean_path(poni_path)),
                                  mask_p, os.path.getmtime(mask_p) if mask_p else 0,
                                  tuple(S["ranges"]), S["ai"], S["mask_total"])
            t0 = time.time()
            hp = fm.ring_median_subtract(corr, rm)
            t_ring = time.time() - t0
            valid = ~S["mask_total"]
            sigma = np.sqrt(np.maximum(fr, 0) + info["scale"] ** 2 * S["ref"].var_of_mean + 1e-9)
            z = corr / sigma
            spot_z = hp / sigma
            spot_z = spot_z / (1.4826 * np.median(np.abs(spot_z[valid])))   # empirical noise units
            _, n_clusters = ndimage.label(spot_z > 5)
            lo_, hi_ = np.percentile(corr[valid], [1, 99])
            mt = S["mask_total"]
            fig, ax = plt.subplots(1, 4, figsize=(30, 7.5))
            ax[0].imshow(np.where(mt, np.nan, fr), origin="lower", vmin=0,
                         vmax=np.percentile(fr[valid], 99.5)); ax[0].set_title("Raw frame")
            ax[1].imshow(np.where(mt, np.nan, corr), origin="lower", vmin=lo_, vmax=hi_,
                         cmap="RdBu_r"); ax[1].set_title("Background-subtracted (1-99 percentile)")
            ax[2].imshow(np.where(mt, np.nan, z), origin="lower", vmin=-6, vmax=6,
                         cmap="RdBu_r"); ax[2].set_title("Significance (sigma)")
            ax[3].imshow(block_max(np.where(mt, 0, spot_z)), origin="lower", vmin=0, vmax=12,
                         cmap="magma", extent=[0, spot_z.shape[1], 0, spot_z.shape[0]])
            ax[3].set_title("Spot view: ring median removed (block max, noise units)")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption(f"{Path(p).name} · scale {info['scale']:.3f} · "
                       f"{int((spot_z > 5).sum())} pixels above 5σ in {n_clusters} clusters in the "
                       f"spot view · subtraction {t_sub:.2f} s, ring-median {t_ring:.2f} s")
        except Exception as e:
            st.error(f"Four-panel view failed: {e}")
    with st.expander("How to read the four panels"):
        st.markdown(
            "1. **Raw**: what the detector recorded.\n"
            "2. **Background-subtracted**: the corrected frame. Ring residuals set the colour "
            "scale, so small spots are easy to miss here. That is expected.\n"
            "3. **Significance**: corrected counts divided by the expected noise.\n"
            "4. **Spot view**: the median of each ring is removed, so anything smooth around a "
            "ring disappears and isolated spots stand out. This is the best panel for judging "
            "spotty hits. A powder ring from fine crystallites is smooth around the ring, so "
            "this view removes it too; judge those from the integrated pattern."
        )

    # ---- batch ----
    st.markdown("#### Run on the whole folder")
    out_dir = out_folder or (sub_folder.rstrip("/\\") + "_corrected_2D" if sub_folder else "")
    xy_dir = (xy_folder_in or (out_dir + "_xy")) if (write_xy and out_dir) else None
    if out_dir:
        st.caption(f"Corrected frames go to: {out_dir}" + (f"  ·  .xy to: {xy_dir}" if xy_dir else ""))
        if sub_folder and Path(out_dir).resolve() == Path(sub_folder).resolve():
            st.error("The output folder is the same as the raw folder. Choose a different folder.")

    same_dir = bool(out_dir and sub_folder and Path(out_dir).resolve() == Path(sub_folder).resolve())
    if st.button("Subtract reference from all frames", type="primary",
                 disabled=(not ready) or same_dir, key="run_btn"):
        try:
            S = build_settings()
            files = sub_files[: int(limit_n)] if limit_n else sub_files
            bar = st.progress(0.0, text="Starting...")
            t0 = time.time()

            def _prog(done, total, name):
                el = time.time() - t0
                eta = el / done * (total - done) if done else 0
                bar.progress(done / total, text=f"{done}/{total}  {name}  "
                                                f"({el / done:.2f} s/file, about {eta / 60:.1f} min left)")

            kw = dict(fit_region=S["fit_region"]) if S["fit_region"] is not None else {}
            rows = fm.process_run(
                files, S["ref"], out_dir, mask=S["mask_total"], ai=S["ai"], xy_dir=xy_dir,
                write_nxs=True, npt=int(npt), rmin=float(rmin), rmax=float(rmax),
                suffix=suffix, interp_ranges=S["ranges"] or None, progress=_prog,
                skip_existing=skip_existing, **kw)
            errors = getattr(fm.process_run, "last_errors", [])
            elapsed = time.time() - t0

            manifest = dict(
                app=APP_VERSION, created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                raw_folder=sub_folder, output_folder=out_dir, xy_folder=xy_dir,
                n_files_requested=len(files), n_frames_written=len(rows), n_errors=len(errors),
                reference_file=ref_use, reference_label=S["ref"].label,
                reference_n_frames=S["ref"].n_used,
                mask_file=clean_path(mask_path), poni_file=clean_path(poni_path),
                extra_mask_2theta_deg=S["ranges"],
                scale_fit=("whole detector" if S["fit_region"] is None else [fit_lo, fit_hi]),
                integration=dict(npt=int(npt), rmin=float(rmin), rmax=float(rmax)) if xy_dir else None,
                suffix=suffix, skip_existing=skip_existing, elapsed_s=round(elapsed, 1),
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(out_dir) / "subtraction_manifest.json", "w") as fh:
                json.dump(manifest, fh, indent=2)
            bar.progress(1.0, text=f"Done: {len(rows)} frames in {elapsed / 60:.1f} min")
            st.session_state["batch_result"] = dict(rows=rows, errors=errors, out_dir=out_dir,
                                                    elapsed=elapsed, n=len(files))
        except Exception as e:
            st.error(f"Run failed: {e}")

    br = st.session_state.get("batch_result")
    if br:
        rows, errors = br["rows"], br["errors"]
        st.success(f"Wrote {len(rows)} corrected frames to {br['out_dir']}")
        if errors:
            st.error(f"{len(errors)} file(s) failed and were skipped (listed in subtraction_errors.csv)")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)
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
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("A scale that drifts or jumps means the intensity changed during the run "
                       "or the reference no longer matches (new solvent, detector moved). "
                       "Flagged frames are worth a look before training on them.")
            st.dataframe(df.drop(columns=["order"]), use_container_width=True, height=250)


# ===========================================================================
# TAB 3: COMPARE WITH 1D  (new in fep2d_app_v3_compare.py; reads .xy files only)
# ===========================================================================
def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _df(df, **kw):
    """st.dataframe stretched to the column width, on old and new Streamlit versions."""
    try:
        return st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        return st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def _pyplot(fig):
    try:
        return st.pyplot(fig, width="stretch")
    except TypeError:
        return _pyplot(fig)



def _fig_png(fig) -> bytes:
    import io as _io
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    return buf.getvalue()


def _save_results(out_dir: Path, results: dict, summary, ref, settings: dict) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, res in results.items():
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key) or "run"
        res["peaks"].to_csv(out_dir / f"{safe}_peaks.csv", index=False)
        (out_dir / f"{safe}_metrics.json").write_text(json.dumps(_json_safe(res["metrics"]), indent=2))
        fig = fc.make_figure(res, ref=ref, label=key)
        fig.savefig(out_dir / f"{safe}_comparison.png", dpi=130)
        plt.close(fig)
        written += [f"{safe}_peaks.csv", f"{safe}_metrics.json", f"{safe}_comparison.png"]
    if summary is not None:
        summary.to_csv(out_dir / "comparison_summary.csv", index=False)
        written.append("comparison_summary.csv")
    (out_dir / "comparison_settings.json").write_text(json.dumps(_json_safe(
        {**settings, "app": APP_VERSION, "written": datetime.datetime.now().isoformat(timespec="seconds")}), indent=2))
    written.append("comparison_settings.json")
    return written


def _show_one(key: str, res: dict, ref):
    st.subheader(key)
    fl = fc.flags(res)
    if fl:
        st.markdown("\n".join(f"- {t}" for t in fl))
    left, right = st.columns([2, 3])
    with left:
        _df(fc.metrics_table(res))
    with right:
        fig = fc.make_figure(res, ref=ref, label=key)
        _pyplot(fig)
        png = _fig_png(fig)
        plt.close(fig)
    st.markdown("**Matched peaks** (heights relative to each pattern's own main peak = 1.00; "
                "2D / 1D below 1 means the peak is smaller in 2D)")
    pk = res["peaks"].copy()
    if len(pk):
        _df(pk.round(3))
    else:
        st.info("No peaks matched. Lower the prominence in Settings, or check both files are the same frames.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Noise and typical level** (fractions of the main peak)")
        _df(res["noise_level"].round(4))
    with c2:
        st.markdown("**Mask-edge windows** (native points; max is in raw counts)")
        _df(res["edges"].round(3))
    st.caption(f"Peaks found only in 1D (deg): {res['unmatched_1d'] or 'none'}.  "
               f"Only in 2D (deg): {res['unmatched_2d'] or 'none'}.")
    d1, d2, d3 = st.columns(3)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key) or "run"
    d1.download_button("Download peaks (.csv)", pk.to_csv(index=False).encode(), f"{safe}_peaks.csv", "text/csv",
                       key=f"dl_pk_{safe}")
    d2.download_button("Download measures (.csv)", fc.metrics_table(res).to_csv(index=False).encode(),
                       f"{safe}_measures.csv", "text/csv", key=f"dl_m_{safe}")
    d3.download_button("Download figure (.png)", png, f"{safe}_comparison.png", "image/png", key=f"dl_f_{safe}")


with tab_cmp:
    st.markdown(
        "Compare the **1D-subtracted** pattern with the **2D-corrected** pattern of the same run: "
        "do the peaks stay in the same places, how do their heights compare, and is the noise or the "
        "baseline different. Each pattern is scaled so its own strongest peak is 1.00. "
        "This tab only reads `.xy` files."
    )
    cmp_mode = st.radio("Compare", ["One run (two .xy files)", "Several runs (two folders of .xy files)"],
                        horizontal=True, key="cmp_mode")
    if cmp_mode.startswith("One"):
        cc1, cc2 = st.columns(2)
        cmp_p1 = clean_path(cc1.text_input("1D-subtracted pattern (.xy)", key="cmp_p1",
                                           placeholder=r"E:/AP39/Run008/1D_BakgRemoval_Run008.xy"))
        cmp_p2 = clean_path(cc2.text_input("2D-corrected pattern (.xy)", key="cmp_p2",
                                           placeholder=r"E:/AP39/Run008/2D_BakgRemoval_Run008.xy"))
    else:
        cc1, cc2 = st.columns(2)
        cmp_d1 = clean_path(cc1.text_input("Folder of 1D .xy files", key="cmp_d1"))
        cmp_d2 = clean_path(cc2.text_input("Folder of 2D .xy files", key="cmp_d2"))
        tc1, tc2, _ = st.columns([1, 1, 3])
        cmp_t1 = tc1.text_input("Tag in 1D file names", value="1D", key="cmp_t1")
        cmp_t2 = tc2.text_input("Tag in 2D file names", value="2D", key="cmp_t2")
        st.caption("Files are paired when their names match once the tag is removed: "
                   "`1D_BakgRemoval_Run007.xy` and `2D_BakgRemoval_Run007.xy` are one pair.")
    cmp_ref = clean_path(st.text_input("Reference pattern for the overlay (optional .xy)", key="cmp_ref",
                                       placeholder=r"E:/refs/Ortho_Dihydrate_FEFNOT05_ref_0.4978.xy"))
    with st.expander("Settings", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        cmp_mask = s1.text_input("Masked range (deg)", value=fc.fmt_ranges([fc.DEFAULTS["mask"]]), key="cmp_mask",
                                 help="Ignored when finding peaks and correlating. Slightly wider than the 2D mask.")
        cmp_prom = s2.number_input("Peak prominence (fraction of main peak)", 0.01, 0.5, fc.DEFAULTS["prominence"],
                                   0.01, key="cmp_prom")
        cmp_tol = s3.number_input("Match tolerance (deg)", 0.01, 0.3, fc.DEFAULTS["match_tol"], 0.01, key="cmp_tol")
        cmp_strong = s4.number_input("'Strong' peak: 1D height above", 0.05, 1.0, fc.DEFAULTS["strong"], 0.05,
                                     key="cmp_strong")
        r1, r2, r3 = st.columns(3)
        cmp_noise = r1.text_input("Noise regions (deg)", value=fc.fmt_ranges(fc.DEFAULTS["noise_regions"]),
                                  key="cmp_noise", help="The first one is also used for main peak / noise.")
        cmp_level = r2.text_input("Level regions (deg)", value=fc.fmt_ranges(fc.DEFAULTS["level_regions"]),
                                  key="cmp_level")
        cmp_edges = r3.text_input("Mask-edge windows (deg)", value=fc.fmt_ranges(fc.DEFAULTS["edge_windows"]),
                                  key="cmp_edges", help="Where to look for a spike or a lost peak at the mask edges.")
        r4, r5 = st.columns(2)
        cmp_corr = r4.text_input("Correlation regions (deg)", value=fc.fmt_ranges(fc.DEFAULTS["corr_regions"]),
                                 key="cmp_corr")
        cmp_hi = r5.number_input("High-angle peaks: above (deg)", 5.0, 40.0, fc.DEFAULTS["high_angle"], 1.0,
                                 key="cmp_hi")
    cmp_out = clean_path(st.text_input("Save results to this folder (optional)", key="cmp_out",
                                       help="Writes the summary .csv, each run's peaks .csv, measures .json and "
                                            "figure .png, and the settings used."))

    if st.button("Compare", type="primary", key="cmp_go"):
        try:
            mk = fc.parse_ranges(cmp_mask)
            if len(mk) != 1:
                raise ValueError("Masked range: give exactly one range, for example 5.2-6.1")
            kw = dict(mask=mk[0], prominence=float(cmp_prom), match_tol=float(cmp_tol), strong=float(cmp_strong),
                      high_angle=float(cmp_hi), noise_regions=tuple(fc.parse_ranges(cmp_noise)),
                      level_regions=tuple(fc.parse_ranges(cmp_level)), edge_windows=tuple(fc.parse_ranges(cmp_edges)),
                      corr_regions=tuple(fc.parse_ranges(cmp_corr)))
            if cmp_mode.startswith("One"):
                if not cmp_p1 or not cmp_p2:
                    raise ValueError("Give both .xy files")
                for pth in (cmp_p1, cmp_p2):
                    if not os.path.isfile(pth):
                        raise FileNotFoundError(f"File not found: {pth}")
                jobs = [(Path(cmp_p2).stem, cmp_p1, cmp_p2)]
            else:
                if not (os.path.isdir(cmp_d1) and os.path.isdir(cmp_d2)):
                    raise FileNotFoundError("Both folders must exist")
                jobs, only1, only2 = fc.find_pairs(cmp_d1, cmp_d2, cmp_t1.strip(), cmp_t2.strip())
                if only1 or only2:
                    st.warning(f"Not paired: {len(only1)} 1D and {len(only2)} 2D file(s) have no partner "
                               f"(1D: {only1[:5]}{'...' if len(only1) > 5 else ''}; 2D: {only2[:5]}{'...' if len(only2) > 5 else ''}).")
                if not jobs:
                    raise ValueError("No pairs found. Check the folders and the tags (the file names must match "
                                     "once the tag is removed).")
            ref = None
            if cmp_ref:
                if not os.path.isfile(cmp_ref):
                    raise FileNotFoundError(f"Reference file not found: {cmp_ref}")
                ref = fc.load_xy(cmp_ref)
            results, rows, errors = {}, [], []
            bar = st.progress(0.0, text="Comparing ...")
            for n, (key, a_path, b_path) in enumerate(jobs):
                try:
                    xa, ya = fc.load_xy(a_path)
                    xb, yb = fc.load_xy(b_path)
                    res = fc.compare(xa, ya, xb, yb, **kw)
                    results[key] = res
                    rows.append(fc.summary_row(key, res))
                except Exception as e:  # keep going: one bad pair must not stop the batch
                    errors.append((key, f"{type(e).__name__}: {e}"))
                bar.progress((n + 1) / len(jobs), text=f"Compared {n + 1} of {len(jobs)}")
            bar.empty()
            summary = pd.DataFrame(rows) if len(rows) > 1 else None
            saved = None
            if cmp_out and results:
                saved = _save_results(Path(cmp_out), results, summary, ref,
                                      {"mode": cmp_mode, "settings": kw,
                                       "pairs": [[k, str(a), str(b)] for k, a, b in jobs],
                                       "reference_overlay": cmp_ref or None})
            st.session_state["cmp"] = dict(results=results, summary=summary, ref=ref, errors=errors,
                                           saved=(cmp_out, saved) if saved else None)
        except Exception as e:
            st.session_state.pop("cmp", None)
            st.error(f"{type(e).__name__}: {e}")

    cmp_state = st.session_state.get("cmp")
    if cmp_state:
        for key, msg in cmp_state["errors"]:
            st.error(f"{key}: {msg}")
        if cmp_state["saved"]:
            st.success(f"Saved {len(cmp_state['saved'][1])} files to {cmp_state['saved'][0]}")
        if cmp_state["summary"] is not None:
            st.subheader(f"Summary: {len(cmp_state['summary'])} runs")
            st.caption("Ratios are 2D / 1D after each pattern is scaled to its own main peak. Edge columns are the "
                       "height inside that window as a fraction of the main peak.")
            _df(cmp_state["summary"].round(3))
            st.download_button("Download summary (.csv)", cmp_state["summary"].to_csv(index=False).encode(),
                               "comparison_summary.csv", "text/csv", key="dl_summary")
        keys = list(cmp_state["results"])
        if keys:
            pick = keys[0] if len(keys) == 1 else st.selectbox("Show detail for", keys, key="cmp_pick")
            _show_one(pick, cmp_state["results"][pick], cmp_state["ref"])
