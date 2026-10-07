"""
fep2d_app_v3.py  -  2D FEP background subtraction (Streamlit)
=============================================================
Tab 1  Build reference : folder of 2D reference frames -> one merged reference (.npz)
Tab 2  Subtract frames : folder of raw frames -> folder of corrected 2D frames
                         (+ optional .xy), with a run manifest for provenance.

Run:   streamlit run fep2d_app_v3.py
Needs: fep2d_v3.py in the same folder, plus numpy, h5py, pyFAI, matplotlib,
       pandas, streamlit.

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

APP_VERSION = "fep2d_app_v3 / fep2d_v3 (V2 maths unchanged)"

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
tab_ref, tab_sub = st.tabs(["1 · Build reference", "2 · Subtract frames"])


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
