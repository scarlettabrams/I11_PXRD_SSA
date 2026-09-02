#Launching the app (local test)
# cd C:\Users\ezxsa27\I11_PXRD_SSA
# conda activate I11_PXRD_SSA_env
# streamlit run app_ssa_V2_deployvariant.py
#
# For Streamlit Community Cloud: point the app at this file with
# requirements_deployvariant.txt in the same repo — no other setup needed.

"""
I11 PXRD Processing Pipeline — CLOUD DEPLOY VARIANT
Streamlit app — Phase 1: inline 04A core loop

This is a cloud-hosted viewer/light-processing variant of app_ssa_V2.py,
for sharing results with colleagues/supervisors remotely. It differs from
the local (beamtime laptop) version in three ways, because a remote cloud
server has no access to your local drives, no display for native GUI
windows, and cannot launch local .exe files:

  1. All "Browse" buttons are file UPLOADERS instead of native OS dialogs.
     Point them at files on whatever machine is viewing the app in a
     browser (your laptop, a colleague's laptop, etc.) — they get copied
     into a temporary folder on the server for that session.
  2. The pyFAI-calib2 GUI launch step is removed — that GUI can't run on
     a headless cloud server. Run calibration locally using the full
     app_ssa_V2.py version, then upload the resulting .poni and mask.npy
     files here in Step 3 of the Calibration tab.
  3. Nothing persists between sessions — the server's storage is
     temporary. Download anything you want to keep using the download
     buttons rather than relying on the output folder paths shown.

Run with: streamlit run app_ssa_V2_deployvariant.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import h5py
import os
import glob
from pathlib import Path
from pybaselines import Baseline
import pyFAI
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


import tempfile


# ── cloud-safe file/folder "picker" ─────────────────────────────────────────────
# Streamlit Cloud runs on a remote headless server: it cannot open a native
# OS dialog (there's no local desktop for it to appear on) and it cannot see
# your laptop's drives. This replacement keeps the SAME function names and
# call signature as the original tkinter version, so every other call site
# in this file (browse_button(...)) is unchanged — but instead of opening a
# dialog, it renders an upload widget and copies whatever is uploaded into a
# temp folder that lives for the duration of the browser session.
def _session_upload_dir(key):
    """One persistent temp directory per session_state key, for this session."""
    dir_key = f"_upload_dir_{key}"
    if dir_key not in st.session_state:
        st.session_state[dir_key] = tempfile.mkdtemp(prefix=f"i11_{key}_")
    return st.session_state[dir_key]


def pick_path(mode="folder", title="Select", filetypes=None, key=None):
    """
    Cloud-compatible stand-in for the local tkinter file/folder dialog.
    Renders a Streamlit uploader and writes the uploaded content into a
    session-scoped temp directory on the server, then stores the resulting
    path in st.session_state[key] — matching the return contract of the
    original tkinter-based version.

    mode : "folder" (multi-file uploader, returns a directory path)
         | "file"   (single-file uploader, returns a file path)
    key  : session_state key to write the result into
    """
    if mode == "folder":
        uploaded = st.file_uploader(
            title, accept_multiple_files=True, key=f"uploader_{key}",
        )
        if not uploaded:
            return None
        target_dir = _session_upload_dir(key)
        for uf in uploaded:
            with open(os.path.join(target_dir, uf.name), "wb") as out:
                out.write(uf.getbuffer())
        if key:
            st.session_state[key] = target_dir
        return target_dir
    else:
        if filetypes:
            exts = ", ".join(ft[1] for ft in filetypes if ft[1] != "*.*")
            if exts:
                st.caption(f"Accepted: {exts}")
        uploaded = st.file_uploader(title, accept_multiple_files=False,
                                     key=f"uploader_{key}")
        if uploaded is None:
            return None
        target_dir = _session_upload_dir(key)
        target_path = os.path.join(target_dir, uploaded.name)
        with open(target_path, "wb") as out:
            out.write(uploaded.getbuffer())
        if key:
            st.session_state[key] = target_path
        return target_path


def browse_button(label, mode, session_key, title="Select", filetypes=None):
    """
    Cloud variant: renders an uploader inline rather than a compact popup
    button, since a remote server can't open a native dialog window. Same
    name/signature as the local app so nothing else needs to change.
    """
    pick_path(mode=mode, title=title, filetypes=filetypes, key=session_key)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="I11 PXRD Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── minimal style ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 20px; border-radius: 6px; }
    div[data-testid="metric-container"] { background: #f8f9fa; padding: 0.6rem 1rem; border-radius: 8px; }
    .status-ok  { color: #1a7a4a; font-weight: 600; }
    .status-err { color: #c0392b; font-weight: 600; }
    .status-warn{ color: #e67e22; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── session state defaults ──────────────────────────────────────────────────────
DEFAULTS = {
    "poni_path": "",
    "mask_path": "",
    "calib_hdf_path": "",
    "calib_verified": False,
    "data_dir": "",
    "run_name": "",
    "half_window": 6,
    "radial_min": 1.0,
    "radial_max": 30.0,
    "npt": 1000,
    "ai": None,
    "mask": None,
    "calib_loaded": False,
    "processed_files": [],
    "last_pattern": None,
    "last_pattern_name": "",
    "output_dir": "",
    "frame_index": 0,
    # ── output folder override (optional; used by all stages) ──
    "global_output_dir": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── helpers ─────────────────────────────────────────────────────────────────────

def load_nxs_frame(filepath):
    """Load a single 2D frame from an I11 .nxs file."""
    with h5py.File(filepath, "r") as f:
        data = np.array(f["/entry1/pixium_hdf/data"][()][:])
    return data.reshape(data.shape[1:]).astype(np.float32)


def integrate_frame(frame, ai, mask, npt, radial_min, radial_max):
    """Azimuthal integration using pyFAI — raw summed counts, no solid angle correction."""
    result = ai.integrate1d(
        frame,
        npt,
        unit=pyFAI.units.TTH_DEG,
        radial_range=[radial_min, radial_max],
        mask=mask,
        correctSolidAngle=False,   # don't divide by solid angle per pixel
        error_model=None,          # no variance scaling
    )
    try:
        return result.radial, result.intensity
    except AttributeError:
        return result[0], result[1]


def baseline_correct(x, y, half_window):
    """MOR baseline correction from pybaselines."""
    fitter = Baseline(x_data=x)
    baseline = fitter.mor(y, half_window=half_window)[0]
    corrected = y - baseline
    return corrected, baseline


def save_xy(path, x, y, header="2theta(deg) Intensity"):
    np.savetxt(path, np.column_stack([x, y]), header=header, comments="", fmt="%.6f")


def load_calib_image(filepath):
    """Load a 2D calibration image from .hdf or .nxs file."""
    with h5py.File(filepath, "r") as f:
        # try known I11 paths first
        for path in ["entry/data/data", "entry1/data/data",
                     "entry1/Pixium10:detector/data", "entry/detector/data"]:
            if path in f:
                data = f[path][()]
                if data.ndim == 3:
                    return data[0].astype(np.float32)
                return data.astype(np.float32)
        # fallback: find any 2D/3D dataset
        found = {}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                found[name] = obj.shape
        f.visititems(visitor)
        if found:
            key = next(iter(found))
            data = f[key][()]
            return (data[0] if data.ndim == 3 else data).astype(np.float32)
    raise RuntimeError(f"No 2D dataset found in {filepath}")


def get_nxs_files(data_dir):
    """Return sorted list of .nxs files in a directory."""
    return sorted(glob.glob(os.path.join(data_dir, "*.nxs")))


def load_1d_pattern_any(filepath, ai=None, mask=None, npt=None,
                         radial_min=None, radial_max=None):
    """
    Load a 1D pattern from either:
      - a .xy file (loaded directly), or
      - a raw 2D frame (.nxs / .hdf) which is integrated on the fly
        using the current calibration (ai, mask, npt, radial range).

    Multi-frame 2D files are mean-averaged across frames first (matches
    the "average several reference frames" convention used elsewhere
    in the pipeline).

    Returns
    -------
    x, y : np.ndarray
    kind : str   "xy" | "raw_frame (N averaged)"
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".xy":
        data = np.loadtxt(filepath, skiprows=1, comments="#")
        if data.ndim < 2 or data.shape[1] < 2:
            raise ValueError(f"{filepath} does not look like a valid 2-column .xy file.")
        return data[:, 0], data[:, 1], "xy"

    # otherwise treat as a raw 2D detector file (.nxs / .hdf)
    if ai is None:
        raise RuntimeError(
            "No calibration loaded — load calibration in the Calibration tab "
            "before using a raw .nxs/.hdf file here (or supply a pre-integrated .xy)."
        )
    with h5py.File(filepath, "r") as f:
        for path in ["entry1/pixium_hdf/data", "entry/data/data", "entry1/data/data",
                     "entry1/Pixium10:detector/data", "entry/detector/data"]:
            if path in f:
                arr = f[path][()]
                break
        else:
            found = {}
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                    found[name] = obj.shape
            f.visititems(visitor)
            if not found:
                raise RuntimeError(f"No 2D dataset found in {filepath}")
            arr = f[next(iter(found))][()]

    n_frames = 1
    if arr.ndim == 3:
        n_frames = arr.shape[0]
        frame = arr.mean(axis=0).astype(np.float32)
    else:
        frame = arr.astype(np.float32)

    x, y = integrate_frame(frame, ai, mask, npt, radial_min, radial_max)
    return x, y, f"raw_frame ({n_frames} averaged)"


def combine_fep_1d(x_s, y_s, x_ref, y_ref, scale_min, scale_max,
                    method="subtract", clip_negative=True, floor_frac=0.02):
    """
    Interpolate a reference pattern onto the sample's 2θ grid and combine
    it with the sample pattern, either by subtraction (physically motivated
    additive-background model) or division (multiplicative correction —
    included for comparison, not as the recommended default; see caveats
    in the FEP tab UI).

    method = "subtract":
        scale_factor = mean(sample)/mean(ref) in [scale_min, scale_max]
        y_combined   = y_s - scale_factor * y_ref_interp   (clipped at 0 if requested)

    method = "divide":
        Reference is floored at floor_frac * ref_max to stop the ratio
        blowing up in low-count tails (a real risk here, since it directly
        amplifies noise). Result is rescaled by the sample's own mean in
        the scale region purely so the output sits on a comparable
        intensity scale to the subtraction case — it is a renormalisation,
        not a physically meaningful "scale factor" the way it is for
        subtraction, and peak heights after division no longer represent
        raw photon counts.

    Returns
    -------
    y_ref_interp   : reference interpolated onto x_s (unscaled, unfloored)
    y_ref_used     : reference as actually used in the combine step
                      (scaled for subtract; floored for divide)
    y_combined     : result of the combine step (optionally clipped at 0)
    factor         : scale_factor (subtract) or renormalisation factor (divide)
    """
    y_ref_interp = np.interp(x_s, x_ref, y_ref)
    region = (x_s >= scale_min) & (x_s <= scale_max)
    if region.sum() < 3:
        raise ValueError(
            "Scaling region contains fewer than 3 points — widen the range "
            "or check your 2θ limits."
        )
    s_region = y_s[region]
    f_region = y_ref_interp[region]

    if method == "divide":
        ref_max = float(y_ref_interp.max())
        floor = max(floor_frac * ref_max, 1e-9)
        y_ref_used = np.clip(y_ref_interp, floor, None)
        ratio = y_s / y_ref_used
        renorm = float(np.mean(s_region)) if np.mean(s_region) > 0 else 1.0
        y_combined = ratio * renorm
        factor = renorm
    else:
        scale_factor = float(np.mean(s_region) / np.mean(f_region)) if np.mean(f_region) > 0 else 1.0
        y_ref_used = y_ref_interp * scale_factor
        y_combined = y_s - y_ref_used
        factor = scale_factor

    if clip_negative:
        y_combined = np.clip(y_combined, 0, None)

    return y_ref_interp, y_ref_used, y_combined, factor


def subtract_fep_1d(x_s, y_s, x_ref, y_ref, scale_min, scale_max, clip_negative=True):
    """Backwards-compatible thin wrapper around combine_fep_1d(method='subtract')."""
    return combine_fep_1d(x_s, y_s, x_ref, y_ref, scale_min, scale_max,
                           method="subtract", clip_negative=clip_negative)


def render_fep_ab_comparison(
    method, key_prefix,
    ab_sample, ab_ref_a, ab_ref_b, ab_ref_a_label, ab_ref_b_label,
    ab_scale_min, ab_scale_max, ab_diag_min, ab_diag_max,
    ab_clip_negative, floor_frac=0.02,
    mask_enabled=False, mask_min=5.4, mask_max=5.7, mask_method="interpolate",
):
    """
    Runs load -> combine (subtract or divide) -> optional intense-peak mask
    for both references and renders the 2-stage plots + diagnostics table +
    save button, inside whatever Streamlit container is currently active
    (i.e. call this from within a `with tab:` block). One call = one method,
    so it can be dropped into separate Subtract / Divide sub-tabs cleanly.

    Baselining is NOT done here — this tool's job ends at FEP removal (+ mask).
    Saved output feeds into the Baselining tab.
    """
    run_ab = st.button(
        f"Run {method} comparison", type="primary", key=f"{key_prefix}_run",
        disabled=not (ab_sample and ab_ref_a and ab_ref_b),
    )
    if not run_ab:
        return

    with st.spinner(f"Loading and {method}ing..."):
        try:
            ai = st.session_state.ai
            mask = st.session_state.mask
            npt = st.session_state.npt
            rmin, rmax = st.session_state.radial_min, st.session_state.radial_max

            x_s, y_s, kind_s = load_1d_pattern_any(ab_sample, ai, mask, npt, rmin, rmax)
            x_a, y_a, kind_a = load_1d_pattern_any(ab_ref_a, ai, mask, npt, rmin, rmax)
            x_b, y_b, kind_b = load_1d_pattern_any(ab_ref_b, ai, mask, npt, rmin, rmax)

            results_ab = {}
            for label, x_r, y_r in [(ab_ref_a_label, x_a, y_a), (ab_ref_b_label, x_b, y_b)]:
                _, y_ref_used, y_comb, factor = combine_fep_1d(
                    x_s, y_s, x_r, y_r, ab_scale_min, ab_scale_max,
                    method=method, clip_negative=ab_clip_negative,
                    floor_frac=floor_frac,
                )
                y_masked = (
                    mask_region(x_s, y_comb, mask_min, mask_max, method=mask_method)
                    if mask_enabled else y_comb
                )

                diag_m = (x_s >= ab_diag_min) & (x_s <= ab_diag_max)
                diag_std = float(y_masked[diag_m].std()) if diag_m.sum() > 2 else float("nan")
                diag_mean = float(y_masked[diag_m].mean()) if diag_m.sum() > 2 else float("nan")

                results_ab[label] = dict(
                    x=x_s, y_ref_used=y_ref_used, y_combined=y_comb, y_masked=y_masked,
                    factor=factor, diag_std=diag_std, diag_mean=diag_mean,
                )

            st.success(f"Sample: {kind_s}  ·  Ref A: {kind_a}  ·  Ref B: {kind_b}  ·  Method: {method}")

            # ── Stage 1: raw integrated ─────────────────────────────
            st.markdown("##### Stage 1 — after integration (raw)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=x_s, y=y_s, mode="lines", name="Sample",
                                       line=dict(color="#2c3e50", width=1.3)))
            fig1.add_trace(go.Scatter(x=x_a, y=y_a, mode="lines", name=ab_ref_a_label,
                                       line=dict(color="#e74c3c", width=1, dash="dash")))
            fig1.add_trace(go.Scatter(x=x_b, y=y_b, mode="lines", name=ab_ref_b_label,
                                       line=dict(color="#2980b9", width=1, dash="dash")))
            fig1.update_layout(
                xaxis_title="2θ (°)", yaxis_title="Intensity",
                xaxis=dict(range=[rmin, rmax]), hovermode="x unified",
                height=380, margin=dict(l=60, r=20, t=20, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig1, use_container_width=True, key=f"{key_prefix}_fig1")

            # ── Stage 2: FEP-combined, masked (final output of this tab) ──
            stage2_verb = "division" if method == "divide" else "subtraction"
            st.markdown(f"##### Stage 2 — after FEP {stage2_verb}" +
                        (" + intense-peak mask" if mask_enabled else "") +
                        " (final output of this tab)")
            fig2 = go.Figure()
            for label, color in [(ab_ref_a_label, "#e74c3c"), (ab_ref_b_label, "#2980b9")]:
                r = results_ab[label]
                fig2.add_trace(go.Scatter(x=r["x"], y=r["y_combined"], mode="lines",
                                           name=f"{label} — {method} (unmasked)",
                                           line=dict(color=color, width=1.0, dash="dot"),
                                           opacity=0.5 if mask_enabled else 1.0))
                if mask_enabled:
                    fig2.add_trace(go.Scatter(x=r["x"], y=r["y_masked"], mode="lines",
                                               name=f"{label} — masked",
                                               line=dict(color=color, width=1.6)))
            fig2.add_vrect(x0=ab_scale_min, x1=ab_scale_max, fillcolor="#f39c12",
                           opacity=0.10, layer="below", line_width=0,
                           annotation_text="scale region", annotation_position="top left",
                           annotation_font_size=10)
            if mask_enabled:
                fig2.add_vrect(x0=mask_min, x1=mask_max, fillcolor="#c0392b",
                               opacity=0.12, layer="below", line_width=0,
                               annotation_text="mask region", annotation_position="top right",
                               annotation_font_size=10)
            fig2.update_layout(
                xaxis_title="2θ (°)", yaxis_title="Intensity",
                xaxis=dict(range=[rmin, rmax]), hovermode="x unified",
                height=420, margin=dict(l=60, r=20, t=20, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_fig2")
            if method == "divide":
                st.caption(
                    "Watch for spikes here — they mark where the reference was weak enough "
                    "to hit the floor and the ratio is unstable, not real signal."
                )

            # ── diagnostics table ────────────────────────────────────
            import pandas as pd
            factor_col = "Renormalisation factor" if method == "divide" else "Scale factor"
            diag_rows = []
            for label in [ab_ref_a_label, ab_ref_b_label]:
                r = results_ab[label]
                diag_rows.append({
                    "Reference": label,
                    "Method": method,
                    factor_col: f"{r['factor']:.4f}",
                    "Diagnostic region mean": f"{r['diag_mean']:.2f}",
                    "Diagnostic region std": f"{r['diag_std']:.2f}",
                    "Max intensity": f"{r['y_masked'].max():.1f}",
                    "Peak 2θ (°)": f"{r['x'][int(np.argmax(r['y_masked']))]:.3f}",
                })
            st.markdown("##### Comparison summary")
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Lower diagnostic-region std/mean = cleaner, less-biased result in a region "
                "with no genuine sample signal. Also compare Stage 2 visually around any "
                "diagnostic peaks you care about (e.g. known polymorph reflections) — a "
                "numerically 'cleaner' reference/method that also removes real peak intensity "
                "is not necessarily the better choice. To test subtract vs divide directly, run "
                "both sub-tabs on the same sample+reference pair and check whether a known "
                "peak (e.g. a confirmed glycine reflection) survives with comparable intensity "
                "in both. This tab does not baseline — take whichever result you settle on to "
                "the Baselining tab next."
            )

            # ── save both patterns (feeds the Baselining tab) ────────
            if st.button("Save both patterns as .xy", key=f"{key_prefix}_save"):
                out_dir = get_stage_dir("02_fep_corrected")
                sample_stem = Path(ab_sample).stem
                for label in [ab_ref_a_label, ab_ref_b_label]:
                    r = results_ab[label]
                    safe_label = "".join(c if c.isalnum() else "_" for c in label)
                    out_xy = os.path.join(out_dir, f"{sample_stem}_FEP{method}_{safe_label}.xy")
                    save_xy(out_xy, r["x"], r["y_masked"],
                            header=f"2theta(deg) Intensity_FEP{method}")
                st.success(f"Saved both patterns to `{out_dir}`")

        except Exception as e:
            st.error(f"{method.capitalize()} comparison failed: {e}")
            st.exception(e)


def get_stage_dir(stage_name):
    """
    Resolve the output folder for a pipeline stage, honouring the sidebar's
    optional global output-folder override. Falls back to the existing
    convention of a folder named "<run_name>_<stage_name>" next to the data
    directory if no override is set. Always creates the folder.

    stage_name examples: "01_integrated", "02_fep_corrected", "03_baselined",
    "04_merged"
    """
    run = st.session_state.run_name or "run"
    override = st.session_state.get("global_output_dir", "")
    base = override if override else st.session_state.data_dir
    out = os.path.join(base, f"{run}_{stage_name}")
    os.makedirs(out, exist_ok=True)
    return out


def mask_region(x, y, mask_min, mask_max, method="interpolate"):
    """
    Knock out an intense residual feature (e.g. an incompletely-removed FEP
    peak) in [mask_min, mask_max] after FEP subtraction/division.

    method = "interpolate": replace the region with a straight line between
        its edge values — keeps the pattern continuous, doesn't introduce a
        false drop to zero, and is generally the safer default for anything
        that will be baselined afterwards.
    method = "zero": set the region to 0.0 directly.

    Returns a copy of y with the region modified; x is unchanged.
    """
    y_out = y.copy()
    region = (x >= mask_min) & (x <= mask_max)
    if not region.any():
        return y_out
    if method == "zero":
        y_out[region] = 0.0
    else:
        idx = np.where(region)[0]
        lo, hi = idx[0], idx[-1]
        lo_val = y_out[lo - 1] if lo > 0 else y_out[hi + 1] if hi + 1 < len(y_out) else 0.0
        hi_val = y_out[hi + 1] if hi + 1 < len(y_out) else lo_val
        y_out[region] = np.linspace(lo_val, hi_val, region.sum())
    return y_out


def render_merge_section(in_glob, out_stage, title="Merge → single PXRD pattern"):
    """
    Averages every .xy file matching in_glob onto a common 2θ grid (the first
    file's grid) and saves the merged pattern + a static PNG into the
    out_stage folder. Shared by the Baselining tab's merge step.
    """
    xy_in = sorted(glob.glob(in_glob))

    if not xy_in:
        st.info(f"No files found matching `{in_glob}` yet.")
        return

    st.markdown(f"**{len(xy_in)}** frames ready to merge.")

    if st.button(title, type="primary", key=f"merge_btn_{out_stage}"):
        with st.spinner("Merging..."):
            try:
                intensities = []
                two_theta_ref = None

                for fpath in xy_in:
                    data = np.loadtxt(fpath, skiprows=1, comments="#")
                    if data.ndim < 2 or data.shape[1] < 2:
                        continue
                    x_i, y_i = data[:, 0], data[:, 1]
                    if two_theta_ref is None:
                        two_theta_ref = x_i
                    y_interp = np.interp(two_theta_ref, x_i, y_i)
                    intensities.append(y_interp)

                if not intensities:
                    st.error("No valid .xy files could be read.")
                    return

                merged = np.mean(intensities, axis=0)
                std_pattern = np.std(intensities, axis=0)

                run = st.session_state.run_name or "run"
                hw = st.session_state.half_window
                merged_stem = f"{run}_hw{hw}_merged_final"
                merged_dir = get_stage_dir(out_stage)
                xy_out = os.path.join(merged_dir, merged_stem + ".xy")
                png_out = os.path.join(merged_dir, merged_stem + ".png")

                save_xy(xy_out, two_theta_ref, merged, header="2theta(deg) Intensity(averaged)")

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(two_theta_ref, merged, color="#1a7a4a",
                        linewidth=1.0, label=f"Merged ({len(intensities)} frames)")
                ax.fill_between(two_theta_ref, merged - std_pattern, merged + std_pattern,
                                 color="#1a7a4a", alpha=0.15, label="±1σ")
                ax.set_xlabel("2θ (°)", fontsize=12)
                ax.set_ylabel("Intensity (a.u.)", fontsize=12)
                ax.set_title(f"{run} — merged PXRD pattern ({len(intensities)} frames)",
                             fontsize=11)
                ax.set_xlim(st.session_state.radial_min, st.session_state.radial_max)
                ax.legend(fontsize=10)
                ax.grid(alpha=0.2)
                fig.tight_layout()
                fig.savefig(png_out, dpi=300, bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

                st.success(f"✓ Merged {len(intensities)} frames\n\nSaved: `{xy_out}`")

                c1, c2, c3 = st.columns(3)
                c1.metric("Frames merged", len(intensities))
                c2.metric("Max intensity", f"{merged.max():.1f}")
                c3.metric("Peak 2θ (°)", f"{two_theta_ref[np.argmax(merged)]:.3f}")

            except Exception as e:
                st.error(f"Merge failed: {e}")
                st.exception(e)


def plot_pattern(x, y, baseline, corrected, title, radial_min, radial_max):
    """Return a matplotlib figure showing raw, baseline, and corrected pattern."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: raw + fitted baseline
    axes[0].plot(x, y, color="#2c3e50", linewidth=0.8, label="Raw integrated")
    axes[0].plot(x, baseline, color="#e74c3c", linewidth=1.2,
                 linestyle="--", label="Baseline (MOR)")
    axes[0].set_xlabel("2θ (°)", fontsize=11)
    axes[0].set_ylabel("Intensity", fontsize=11)
    axes[0].set_title("Raw + baseline fit", fontsize=11)
    axes[0].set_xlim(radial_min, radial_max)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)

    # Right: baseline-corrected
    axes[1].plot(x, corrected, color="#1a7a4a", linewidth=1.0, label="Corrected")
    axes[1].axhline(0, color="#aaa", linewidth=0.5, linestyle=":")
    axes[1].set_xlabel("2θ (°)", fontsize=11)
    axes[1].set_ylabel("Intensity", fontsize=11)
    axes[1].set_title(f"Baseline-corrected: {title}", fontsize=11)
    axes[1].set_xlim(radial_min, radial_max)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — configuration (set once per beamtime)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Beamtime setup")
    st.caption("Set these once at the start of a beamtime session.")

    st.markdown("### Calibration")
    if st.session_state.calib_loaded:
        st.markdown('<p class="status-ok">● Calibration ready</p>', unsafe_allow_html=True)
        st.caption(f"PONI: `{Path(st.session_state.poni_path).name}`")
        if st.session_state.mask_path:
            st.caption(f"Mask: `{Path(st.session_state.mask_path).name}`")
    else:
        st.markdown('<p class="status-warn">● No calibration loaded</p>', unsafe_allow_html=True)
        st.caption("Go to the **🎯 Calibration** tab to set up pyFAI and load your .poni file.")

    st.divider()

    st.markdown("### Data location")
    st.caption(
        "Upload your `.nxs` frames below — cloud has no access to local drives, "
        "so typing a path here won't work; use the uploader instead."
    )
    col_dd, col_db = st.columns([3, 1])
    with col_dd:
        data_dir_input = st.text_input(
            "Directory containing .nxs files (auto-filled after upload)",
            value=st.session_state.get("_browse_data_dir", st.session_state.data_dir),
            placeholder="Use the uploader → to add frames",
            help="Populated automatically once you upload files on the right.",
            key="_data_dir_text",
            disabled=True,
        )
    with col_db:
        browse_button("Upload", "folder", "_browse_data_dir", "Upload .nxs frames")
    # sync browsed value into text box on next rerun
    if st.session_state.get("_browse_data_dir"):
        data_dir_input = st.session_state["_browse_data_dir"]
    run_name_input = st.text_input(
        "Run name",
        value=st.session_state.run_name,
        placeholder="Run1_GLY_0.5VF_X2",
        help="Used to name output files and folders",
    )
    if st.button("Set data directory", use_container_width=True):
        if not os.path.isdir(data_dir_input):
            st.error("Directory not found. Check the path.")
        elif not run_name_input.strip():
            st.error("Please enter a run name.")
        else:
            st.session_state.data_dir = data_dir_input
            st.session_state.run_name = run_name_input.strip()
            st.session_state.processed_files = []
            nxs = get_nxs_files(data_dir_input)
            st.success(
                f"Found {len(nxs)} .nxs files. Stage folders will be created as "
                f"`{run_name_input.strip()}_01_integrated`, `..._02_fep_corrected`, "
                f"`..._03_baselined`, `..._04_merged` "
                f"(under {st.session_state.global_output_dir or data_dir_input})."
            )

    st.divider()

    st.markdown("### Processing parameters")
    st.session_state.half_window = st.slider(
        "Baseline half_window",
        min_value=2, max_value=30,
        value=st.session_state.half_window,
        help="Controls baseline smoothing. 5–10 typical for sharp PXRD peaks.",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.radial_min = st.number_input(
            "2θ min (°)", value=st.session_state.radial_min,
            min_value=0.0, max_value=10.0, step=0.5,
        )
    with col2:
        st.session_state.radial_max = st.number_input(
            "2θ max (°)", value=st.session_state.radial_max,
            min_value=10.0, max_value=60.0, step=1.0,
        )
    st.session_state.npt = st.number_input(
        "Integration points (npt)", value=st.session_state.npt,
        min_value=100, max_value=5000, step=100,
    )

    st.divider()

    st.markdown("### Output folder")
    st.caption("Leave blank on cloud — outputs are stored in a temporary session folder automatically.")
    col_od, col_odb = st.columns([3, 1])
    with col_od:
        output_override_input = st.text_input(
            "Override output folder (optional)",
            value=st.session_state.get("_browse_global_output_dir",
                  st.session_state.global_output_dir),
            placeholder=r"E:/I11BT_.../Data_Processing/App_Output",
            help=(
                "If set, every pipeline stage (integrated / FEP-corrected / baselined / "
                "merged) writes into subfolders here instead of next to the raw data. "
                "Leave blank to use the default per-run folders next to your data directory."
            ),
        )
    with col_odb:
        st.markdown("<br>", unsafe_allow_html=True)
        browse_button("Browse", "folder", "_browse_global_output_dir", "Select output folder")
    if st.session_state.get("_browse_global_output_dir"):
        output_override_input = st.session_state["_browse_global_output_dir"]
    st.session_state.global_output_dir = output_override_input.strip()
    if st.session_state.global_output_dir:
        st.caption(f"✓ All stages → `{st.session_state.global_output_dir}/<run>_<stage>/`")
    else:
        st.caption("Default: stage folders created next to the data directory.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — tabs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🎯 I11 PXRD inline pipeline")
st.caption("Diamond Light Source · Beamline I11 · Flow crystallisation")
st.info(
    "☁️ **Cloud viewer build.** Use the upload widgets in place of the old "
    "'Browse' dialogs. Calibration must be run locally (see Calibration tab) "
    "and the .poni/mask.npy uploaded here. Nothing persists between sessions — "
    "use the download options to keep anything you generate.",
    icon="☁️",
)

tab_calib, tab_inline, tab_batch, tab_fep, tab_baseline, tab_viewer, tab_status = st.tabs([
    "🎯  Calibration",
    "▶  Inline loop",
    "⚡  Batch process",
    "🧪  FEP subtraction",
    "📉  Baselining",
    "📊  Pattern viewer",
    "ℹ️  Status",
])

# ── TAB 0: CALIBRATION ────────────────────────────────────────────────────────
with tab_calib:
    st.markdown("### Calibration — start of beamtime")
    st.caption(
        "Complete this once per beamtime window before collecting data. "
        "Load your calibration HDF file, launch pyFAI to pick rings and save "
        "your .poni and mask, then verify here before proceeding."
    )

    st.markdown("#### Step 1 — Load calibration image")
    col_hdf, col_hdfb = st.columns([3, 1])
    with col_hdf:
        calib_hdf_input = st.text_input(
            "Path to calibration .hdf file",
            value=st.session_state.get("_browse_calib_hdf", st.session_state.calib_hdf_path),
            placeholder=r"E:/calib/pixium_122554.hdf",
            help="The CeO2 (or equivalent) calibration frame collected at the beamline.",
        )
    with col_hdfb:
        st.markdown("<br>", unsafe_allow_html=True)
        browse_button("Browse", "file", "_browse_calib_hdf", "Select calibration HDF file",
                      filetypes=[("HDF/NXS files", "*.hdf *.nxs"), ("All files", "*.*")])
    if st.session_state.get("_browse_calib_hdf"):
        calib_hdf_input = st.session_state["_browse_calib_hdf"]

    col_load, col_gap = st.columns([1, 3])
    with col_load:
        load_calib_img_btn = st.button("Load image", use_container_width=True)

    if load_calib_img_btn:
        if not os.path.exists(calib_hdf_input):
            st.error(f"File not found:\n{calib_hdf_input}")
        else:
            try:
                img = load_calib_image(calib_hdf_input)
                st.session_state.calib_hdf_path = calib_hdf_input
                st.success(f"Loaded — shape: {img.shape[0]} × {img.shape[1]} px")

                fig, ax = plt.subplots(figsize=(6, 6))
                vmax = np.percentile(img, 99)
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, origin="lower")
                ax.set_title("Calibration image preview", fontsize=11)
                ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Could not load image: {e}")

    st.divider()
    st.markdown("#### Step 2 — Run pyFAI calibration (do this locally)")
    st.warning(
        "The pyFAI calibration GUI can't run on this cloud server — it needs a "
        "local display and launches a native `.exe`, neither of which exist on "
        "a remote host. Run this step on your own laptop using the full "
        "**app_ssa_V2.py** version (or `pyFAI-calib2` directly), save your "
        "**.poni** file and **mask.npy**, then upload them below in Step 3."
    )
    st.caption(
        "In pyFAI locally: set wavelength (~0.826 Å for I11), select CeO2 calibrant, "
        "apply your detector mask, pick rings, fit geometry, then save the .poni "
        "file and export the mask as .npy."
    )

    st.divider()
    st.markdown("#### Step 3 — Load and verify saved calibration")
    st.caption(
        "Point to the .poni and mask files saved by pyFAI. "
        "A test integration will confirm everything is working before you start collecting."
    )

    col3a, col3b = st.columns(2)
    with col3a:
        col3a1, col3a2 = st.columns([3, 1])
        with col3a1:
            poni_verify = st.text_input(
                "Path to saved .poni file",
                value=st.session_state.get("_browse_poni", st.session_state.poni_path),
                placeholder=r"E:/calib/i11_calib.poni",
                key="poni_verify_input",
            )
        with col3a2:
            st.markdown("<br>", unsafe_allow_html=True)
            browse_button("Browse", "file", "_browse_poni", "Select .poni file",
                          filetypes=[("PONI files", "*.poni"), ("All files", "*.*")])
        if st.session_state.get("_browse_poni"):
            poni_verify = st.session_state["_browse_poni"]

    with col3b:
        col3b1, col3b2 = st.columns([3, 1])
        with col3b1:
            mask_verify = st.text_input(
                "Path to saved mask.npy",
                value=st.session_state.get("_browse_mask", st.session_state.mask_path),
                placeholder=r"E:/calib/mask.npy",
                key="mask_verify_input",
            )
        with col3b2:
            st.markdown("<br>", unsafe_allow_html=True)
            browse_button("Browse", "file", "_browse_mask", "Select mask file",
                          filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
        if st.session_state.get("_browse_mask"):
            mask_verify = st.session_state["_browse_mask"]

    wavelength_input = st.number_input(
        "Wavelength (Å) — for display only",
        value=0.82600, min_value=0.1, max_value=2.5, step=0.001, format="%.5f",
        help="Typical I11 wavelength ≈ 0.826 Å. Confirm with local contact.",
    )

    if st.button("Load and verify calibration", type="primary"):
        ok = True
        if not os.path.exists(poni_verify):
            st.error(f".poni file not found:\n{poni_verify}")
            ok = False
        if mask_verify and not os.path.exists(mask_verify):
            st.error(f"Mask file not found:\n{mask_verify}")
            ok = False

        if ok:
            try:
                ai = pyFAI.load(poni_verify)
                mask = np.load(mask_verify) if mask_verify else None

                # store in session — ready to use in all other tabs
                st.session_state.ai = ai
                st.session_state.mask = mask
                st.session_state.poni_path = poni_verify
                st.session_state.mask_path = mask_verify
                st.session_state.calib_loaded = True
                st.session_state.calib_verified = True

                st.success("✓ Calibration loaded and stored — all tabs are now ready.")

                # show calibration parameters
                st.markdown("**Calibration parameters:**")
                try:
                    st.markdown(f"""
| Parameter | Value |
|---|---|
| Distance (m) | `{ai.dist:.5f}` |
| poni1 (m) | `{ai.poni1:.5f}` |
| poni2 (m) | `{ai.poni2:.5f}` |
| Wavelength (Å) | `{ai.wavelength*1e10:.5f}` |
| Pixel size (μm) | `{ai.pixel1*1e6:.1f} × {ai.pixel2*1e6:.1f}` |
""")
                except Exception:
                    pass

                # run a test integration on the calibration image if available
                if st.session_state.calib_hdf_path and os.path.exists(
                    st.session_state.calib_hdf_path
                ):
                    st.markdown("**Verification — test integration on calibration image:**")
                    with st.spinner("Integrating..."):
                        img = load_calib_image(st.session_state.calib_hdf_path)
                        x_c, y_c = integrate_frame(img, ai, mask, 1500, 1.0, 30.0)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(x_c, y_c, color="#2c3e50", linewidth=0.8)
                        ax.set_xlabel("2θ (°)", fontsize=11)
                        ax.set_ylabel("Intensity", fontsize=11)
                        ax.set_title(
                            "Test integration — CeO₂ peaks should be visible and sharp",
                            fontsize=11
                        )
                        ax.set_xlim(1, 30)
                        ax.grid(alpha=0.2)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            "If peaks look broad, off-position, or missing, "
                            "re-run pyFAI calibration and check ring fitting."
                        )
                else:
                    st.info(
                        "Load a calibration image in Step 1 to also run a "
                        "test integration here."
                    )

            except Exception as e:
                st.error(f"Failed to load calibration: {e}")
                st.session_state.calib_loaded = False
                st.session_state.calib_verified = False


# ── TAB 1: INLINE LOOP ────────────────────────────────────────────────────────
with tab_inline:
    st.markdown("### Near-inline processing — one frame at a time")
    st.caption(
        "Integrates raw frames only — no FEP correction, no baselining. Step through "
        "with Prev / Next, or pick from the dropdown. Use during active data collection. "
        "FEP correction and baselining happen afterwards, in their own tabs."
    )

    if not st.session_state.calib_loaded:
        st.warning("Load your calibration files in the Calibration tab before processing.")
    elif not st.session_state.data_dir:
        st.warning("Set your data directory in the sidebar.")
    else:
        nxs_files = get_nxs_files(st.session_state.data_dir)

        if not nxs_files:
            st.info(f"No .nxs files found in {st.session_state.data_dir}")
        else:
            n_frames = len(nxs_files)

            # clamp index in case file count changed
            if st.session_state.frame_index >= n_frames:
                st.session_state.frame_index = n_frames - 1

            # ── navigation row ──────────────────────────────────────────────
            col_prev, col_idx, col_next, col_latest = st.columns([1, 2, 1, 1])

            with col_prev:
                if st.button("◀ Prev", use_container_width=True,
                             disabled=st.session_state.frame_index == 0):
                    st.session_state.frame_index -= 1
                    st.rerun()

            with col_idx:
                # dropdown synced to frame_index
                chosen = st.selectbox(
                    "Frame",
                    options=list(range(n_frames)),
                    index=st.session_state.frame_index,
                    format_func=lambda i: f"{i+1}/{n_frames}  {os.path.basename(nxs_files[i])}",
                    label_visibility="collapsed",
                )
                if chosen != st.session_state.frame_index:
                    st.session_state.frame_index = chosen
                    st.rerun()

            with col_next:
                if st.button("Next ▶", use_container_width=True,
                             disabled=st.session_state.frame_index == n_frames - 1):
                    st.session_state.frame_index += 1
                    st.rerun()

            with col_latest:
                if st.button("⏭ Latest", use_container_width=True,
                             help="Jump to the most recently collected frame"):
                    st.session_state.frame_index = n_frames - 1
                    st.rerun()

            selected_file = nxs_files[st.session_state.frame_index]
            frame_name = Path(selected_file).stem
            already_done = selected_file in st.session_state.processed_files

            # status line
            status_icon = "✓" if already_done else "○"
            st.caption(
                f"{status_icon} Frame {st.session_state.frame_index + 1} of {n_frames} · "
                f"{frame_name} · "
                f"{'already processed' if already_done else 'not yet processed'} · "
                f"{len(st.session_state.processed_files)} done this session"
            )

            # ── process button ──────────────────────────────────────────────
            process_btn = st.button(
                "Integrate this frame →",
                type="primary",
                use_container_width=False,
            )

            if process_btn:
                with st.spinner(f"Integrating {frame_name}..."):
                    try:
                        frame = load_nxs_frame(selected_file)
                        x, y = integrate_frame(
                            frame,
                            st.session_state.ai,
                            st.session_state.mask,
                            st.session_state.npt,
                            st.session_state.radial_min,
                            st.session_state.radial_max,
                        )
                        out_dir = get_stage_dir("01_integrated")
                        xy_path = os.path.join(out_dir, f"{frame_name}_integrated.xy")
                        save_xy(xy_path, x, y, header="2theta(deg) Intensity_raw")

                        st.session_state.last_pattern = (x, y, None, None)
                        st.session_state.last_pattern_name = frame_name
                        if selected_file not in st.session_state.processed_files:
                            st.session_state.processed_files.append(selected_file)

                        st.success(f"✓ Saved: {os.path.basename(xy_path)}")

                        # interactive plotly chart
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(
                            x=x, y=y, mode="lines", name="Raw integrated",
                            line=dict(color="#2c3e50", width=1.3),
                        ))
                        fig_p.update_layout(
                            title=f"{frame_name} — raw integrated",
                            xaxis_title="2θ (°)",
                            yaxis_title="Intensity",
                            xaxis=dict(
                                range=[st.session_state.radial_min,
                                       st.session_state.radial_max],
                                tickfont=dict(color="black"),
                                title_font=dict(color="black"),
                                linecolor="black",
                                gridcolor="#e0e0e0",
                            ),
                            yaxis=dict(
                                tickfont=dict(color="black"),
                                title_font=dict(color="black"),
                                linecolor="black",
                                gridcolor="#e0e0e0",
                            ),
                            font=dict(color="black"),
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, xanchor="right", x=1,
                                        font=dict(color="black")),
                            height=420,
                            margin=dict(l=60, r=20, t=60, b=50),
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                        # quick stats
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Max intensity", f"{y.max():.1f}")
                        c2.metric("Peak 2θ (°)", f"{x[np.argmax(y)]:.3f}")
                        c3.metric("Frames processed", len(st.session_state.processed_files))

                        # auto-advance hint
                        if st.session_state.frame_index < n_frames - 1:
                            st.info("Click **Next ▶** to move to the next frame.")

                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                        st.exception(e)




# ── TAB 2: BATCH PROCESS ─────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch integration — all frames in directory")
    st.caption(
        "Integrates every unprocessed .nxs frame in the data directory at once — no FEP "
        "correction, no baselining. Use this at the end of a run, or to catch up after a gap. "
        "FEP correction and baselining happen afterwards, in their own tabs."
    )

    if not st.session_state.calib_loaded:
        st.warning("Load your calibration files in the sidebar first.")
    elif not st.session_state.data_dir:
        st.warning("Set your data directory in the sidebar first.")
    else:
        nxs_files = get_nxs_files(st.session_state.data_dir)
        already_done = set(st.session_state.processed_files)
        to_process = [f for f in nxs_files if f not in already_done]

        st.info(
            f"**{len(nxs_files)}** total frames · "
            f"**{len(already_done)}** already processed · "
            f"**{len(to_process)}** to process"
        )

        skip_processed = st.checkbox(
            "Skip already-processed frames", value=True,
            help="Uncheck to reprocess everything from scratch."
        )

        if st.button("Run batch integration", type="primary"):
            target = to_process if skip_processed else nxs_files
            if not target:
                st.info("Nothing to process.")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                results = []
                out_dir = get_stage_dir("01_integrated")

                for i, filepath in enumerate(target):
                    frame_name = Path(filepath).stem
                    status_text.text(f"Integrating {i+1}/{len(target)}: {frame_name}")
                    try:
                        frame = load_nxs_frame(filepath)
                        x, y = integrate_frame(
                            frame,
                            st.session_state.ai,
                            st.session_state.mask,
                            st.session_state.npt,
                            st.session_state.radial_min,
                            st.session_state.radial_max,
                        )
                        xy_path = os.path.join(out_dir, f"{frame_name}_integrated.xy")
                        save_xy(xy_path, x, y, header="2theta(deg) Intensity_raw")
                        results.append({"file": frame_name, "status": "✓ OK",
                                        "max_intensity": f"{y.max():.1f}"})
                        if filepath not in st.session_state.processed_files:
                            st.session_state.processed_files.append(filepath)
                    except Exception as e:
                        results.append({"file": frame_name, "status": f"✗ {e}",
                                        "max_intensity": "—"})

                    progress.progress((i + 1) / len(target))

                status_text.text("Complete.")
                st.success(f"Integrated {len(target)} frames → {out_dir}")

                import pandas as pd
                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Next: use the **FEP subtraction** tab to remove FEP background from "
                    "these integrated patterns, then the **Baselining** tab to morphologically "
                    "baseline and merge them into a final run pattern."
                )


# ── TAB 3: PATTERN VIEWER ─────────────────────────────────────────────────────
with tab_viewer:
    st.markdown("### Pattern viewer — interactive comparison")
    st.caption(
        "Zoom, pan, and hover over patterns. Click legend entries to toggle traces. "
        "Select patterns below to overlay them."
    )

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        _viewer_default = (
            get_stage_dir("04_merged") if st.session_state.data_dir else ""
        )
        xy_dir = st.text_input(
            "Folder containing .xy files",
            value=st.session_state.get("_browse_xy_dir", _viewer_default),
            placeholder=r"E:/beamtime/RAW/Run1_04_merged",
            help="Defaults to the merged-pattern folder — browse to any stage folder "
                 "(01_integrated, 02_fep_corrected, 03_baselined, 04_merged) to compare "
                 "patterns at other points in the pipeline.",
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        browse_button("Browse", "folder", "_browse_xy_dir", "Select folder with .xy files")
    if st.session_state.get("_browse_xy_dir"):
        xy_dir = st.session_state["_browse_xy_dir"]
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("Scan folder", use_container_width=True)

    xy_files = []
    if xy_dir and os.path.isdir(xy_dir):
        xy_files = sorted(glob.glob(os.path.join(xy_dir, "*.xy")))

    if load_btn and not xy_files:
        st.warning("No .xy files found in that folder.")

    # fallback: show last inline pattern if no folder loaded
    if not xy_files and st.session_state.last_pattern is not None:
        st.info("Showing last inline-processed pattern — scan a folder above to load .xy files.")
        x, y, baseline, corrected = st.session_state.last_pattern
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=x, y=corrected, mode="lines",
            name=st.session_state.last_pattern_name,
            line=dict(color="#1a7a4a", width=1.2),
        ))
        fig_v.update_layout(
            xaxis_title="2θ (°)", yaxis_title="Intensity (a.u.)",
            xaxis=dict(range=[st.session_state.radial_min, st.session_state.radial_max]),
            hovermode="x unified", height=480,
            margin=dict(l=60, r=20, t=40, b=50),
        )
        st.plotly_chart(fig_v, use_container_width=True)

    elif xy_files:
        selected_xy = st.multiselect(
            "Select patterns to display",
            options=xy_files,
            default=xy_files[-1:],
            format_func=os.path.basename,
        )

        # ── display options ─────────────────────────────────────────────────
        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            normalise = st.checkbox("Normalise to max = 1", value=False,
                                    help="Scale each pattern independently for shape comparison.")
        with col_opt2:
            offset_mode = st.checkbox("Stack with offset", value=False,
                                      help="Add a vertical offset between patterns for clarity.")
        with col_opt3:
            offset_val = st.number_input("Offset amount", value=1.0, step=0.1,
                                          disabled=not offset_mode,
                                          help="Added between each pattern when stacking.")
        with col_opt4:
            scale_weak = st.number_input(
                "Boost weak patterns (×)", value=1.0, step=0.5, min_value=0.1,
                help="Multiply ALL patterns by this factor before plotting. "
                     "Useful when experimental peaks are very weak vs references."
            )

        # ── plot style expander ─────────────────────────────────────────────
        with st.expander("🎨 Plot style", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                font_size = st.slider("Font size", min_value=8, max_value=24,
                                      value=13, step=1,
                                      help="Applies to axis labels, tick labels, and legend.")
                line_width = st.slider("Line width", min_value=1, max_value=5,
                                       value=1, step=1)
            with sc2:
                show_grid = st.checkbox("Show gridlines", value=True)
                show_axis_lines = st.checkbox("Show axis lines", value=True)
                show_zero_line = st.checkbox("Show zero line (y)", value=False)
            with sc3:
                bg_color = st.selectbox(
                    "Background",
                    options=["white", "transparent (app default)", "#f8f9fa (light grey)"],
                    index=0,
                )
                plot_height = st.slider("Plot height (px)", min_value=300,
                                        max_value=900, value=500, step=50)

            # resolve background string to a colour value
            bg_map = {
                "white": "white",
                "transparent (app default)": "rgba(0,0,0,0)",
                "#f8f9fa (light grey)": "#f8f9fa",
            }
            bg_val = bg_map[bg_color]

        if selected_xy:
            fig_v = go.Figure()
            colors = [
                "#1a7a4a", "#2980b9", "#8e44ad", "#c0392b",
                "#e67e22", "#16a085", "#2c3e50", "#f39c12",
            ]

            for i, fpath in enumerate(selected_xy):
                try:
                    data = np.loadtxt(fpath, skiprows=1, comments="#")
                    if data.ndim < 2 or data.shape[1] < 2:
                        continue
                    xv, yv = data[:, 0], data[:, 1]

                    # clip negatives before normalising — avoids noise spikes
                    # dominating the max and squashing real peaks to zero
                    # (matches notebook 07 behaviour)
                    yv = np.clip(yv, 0, None)

                    if normalise and yv.max() > 0:
                        yv = yv / yv.max()
                    if scale_weak != 1.0:
                        yv = yv * scale_weak
                    if offset_mode:
                        yv = yv + i * offset_val
                    col = colors[i % len(colors)]
                    fig_v.add_trace(go.Scatter(
                        x=xv, y=yv,
                        mode="lines",
                        name=Path(fpath).stem,
                        line=dict(color=col, width=line_width),
                        hovertemplate="2θ: %{x:.3f}°<br>I: %{y:.1f}<extra>%{fullData.name}</extra>",
                    ))
                except Exception as e:
                    st.warning(f"Could not read {os.path.basename(fpath)}: {e}")

            fig_v.update_layout(
                xaxis_title="2θ (°)",
                yaxis_title="Intensity (a.u.)" if not normalise else "Normalised intensity",
                xaxis=dict(
                    range=[st.session_state.radial_min, st.session_state.radial_max],
                    showline=show_axis_lines,
                    showgrid=show_grid,
                    gridcolor="#e0e0e0",
                    linecolor="black",
                    mirror=show_axis_lines,
                    tickfont=dict(size=font_size, color="black"),
                    title_font=dict(size=font_size, color="black"),
                    zeroline=False,
                ),
                yaxis=dict(
                    showline=show_axis_lines,
                    showgrid=show_grid,
                    gridcolor="#e0e0e0",
                    linecolor="black",
                    mirror=show_axis_lines,
                    zeroline=show_zero_line,
                    zerolinecolor="#aaaaaa",
                    zerolinewidth=1,
                    tickfont=dict(size=font_size, color="black"),
                    title_font=dict(size=font_size, color="black"),
                ),
                legend=dict(
                    orientation="v", x=1.01, y=1, xanchor="left",
                    font=dict(size=font_size - 1, color="black"),
                ),
                font=dict(size=font_size, color="black"),
                hovermode="x unified",
                paper_bgcolor=bg_val,
                plot_bgcolor="white" if bg_val != "rgba(0,0,0,0)" else "white",
                height=plot_height,
                margin=dict(l=60, r=180, t=40, b=50),
            )
            st.plotly_chart(fig_v, use_container_width=True)
            st.caption(
                "Tip: click a pattern name in the legend to hide/show it · "
                "double-click to isolate · scroll to zoom · drag to pan"
            )


# ── TAB 5: FEP SUBTRACTION ────────────────────────────────────────────────────
with tab_fep:
    st.markdown("### FEP subtraction")
    st.caption(
        "Remove FEP tubing scattering contributions from integrated 1D patterns. Load "
        "candidate references, decide subtract vs divide, optionally mask an intense "
        "residual FEP peak, then either save a single test pattern or run the whole "
        "batch of integrated frames through the chosen settings. Baselining happens "
        "afterwards, in its own tab."
    )

    # ── method toggle ──────────────────────────────────────────────────────────
    fep_method = st.radio(
        "Subtraction method",
        options=["1D — FEP subtraction (available)", "2D — pre-integration (coming soon)"],
        horizontal=True,
        help=(
            "1D: subtract or divide a scaled FEP reference from the integrated 1D pattern. "
            "Order is integrate → combine → (optional mask) — no baselining here. "
            "2D: subtract FEP in 2D before azimuthal integration (future)."
        ),
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 1D METHOD
    # ══════════════════════════════════════════════════════════════════════════
    if fep_method.startswith("1D"):

        st.markdown("#### Decide the method — reference comparison")
        st.caption(
            "Runs one sample through two candidate FEP references. Accepts either a raw "
            ".nxs/.hdf frame (integrated on the fly using the loaded calibration — "
            "multi-frame files are averaged) or a pre-integrated .xy file, for the sample "
            "and for each reference."
        )

        if not st.session_state.calib_loaded:
            st.warning(
                "No calibration loaded — you can still compare pre-integrated .xy files, "
                "but raw .nxs/.hdf inputs need calibration loaded in the Calibration tab first."
            )

        col_smp, col_refa, col_refb = st.columns(3)
        with col_smp:
            col1_, col2_ = st.columns([3, 1])
            with col1_:
                ab_sample = st.text_input(
                    "Sample (frame or .xy)",
                    value=st.session_state.get("_browse_ab_sample", ""),
                    placeholder=r"E:/.../i11-1-123040.nxs  or  ..._integrated.xy",
                    key="ab_sample_input",
                )
            with col2_:
                st.markdown("<br>", unsafe_allow_html=True)
                browse_button("Browse", "file", "_browse_ab_sample", "Select sample file")
            if st.session_state.get("_browse_ab_sample"):
                ab_sample = st.session_state["_browse_ab_sample"]

        with col_refa:
            col1_, col2_ = st.columns([3, 1])
            with col1_:
                ab_ref_a = st.text_input(
                    "Reference A — e.g. raw FEP tube",
                    value=st.session_state.get("_browse_ab_ref_a", ""),
                    placeholder=r"E:/calib/FEP_empty_ref.nxs  or  .xy",
                    key="ab_ref_a_input",
                )
            with col2_:
                st.markdown("<br>", unsafe_allow_html=True)
                browse_button("Browse", "file", "_browse_ab_ref_a", "Select Reference A")
            if st.session_state.get("_browse_ab_ref_a"):
                ab_ref_a = st.session_state["_browse_ab_ref_a"]
            ab_ref_a_label = st.text_input("Label A", value="Raw FEP ref", key="ab_ref_a_label")

        with col_refb:
            col1_, col2_ = st.columns([3, 1])
            with col1_:
                ab_ref_b = st.text_input(
                    "Reference B — e.g. tri-segmented",
                    value=st.session_state.get("_browse_ab_ref_b", ""),
                    placeholder=r"E:/calib/Water_tri_ref.nxs  or  .xy",
                    key="ab_ref_b_input",
                )
            with col2_:
                st.markdown("<br>", unsafe_allow_html=True)
                browse_button("Browse", "file", "_browse_ab_ref_b", "Select Reference B")
            if st.session_state.get("_browse_ab_ref_b"):
                ab_ref_b = st.session_state["_browse_ab_ref_b"]
            ab_ref_b_label = st.text_input("Label B", value="Tri-segmented ref", key="ab_ref_b_label")

        st.markdown("**Scaling region** — 2θ range used to match each reference to the sample")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            ab_scale_min = st.number_input(
                "Scale region min (°)", value=2.0, min_value=0.0, max_value=10.0, step=0.5,
                key="ab_scale_min",
                help="Pick a region where only FEP scattering is present, no sample peaks.",
            )
        with col_r2:
            ab_scale_max = st.number_input(
                "Scale region max (°)", value=4.0, min_value=0.0, max_value=15.0, step=0.5,
                key="ab_scale_max",
            )

        st.markdown(
            "**Diagnostic region** — an independent 2θ window expected to be flat/featureless "
            "after FEP removal; used to score residual noise/bias for each reference"
        )
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            ab_diag_min = st.number_input(
                "Diagnostic region min (°)", value=7.0, min_value=0.0, max_value=30.0, step=0.5,
                key="ab_diag_min",
            )
        with col_d2:
            ab_diag_max = st.number_input(
                "Diagnostic region max (°)", value=9.0, min_value=0.0, max_value=40.0, step=0.5,
                key="ab_diag_max",
            )

        ab_clip_negative = st.checkbox(
            "Clip negative values to zero after combine step", value=True,
            key="ab_clip_negative",
        )

        st.divider()
        st.markdown("**Intense-peak mask** — optional, applied after the combine step")
        st.caption(
            "For knocking out a residual FEP feature that scaling alone doesn't fully "
            "remove (e.g. a sharp peak around 5.5°). Interpolate draws a straight line "
            "between the region's edges — safer for whatever comes after (baselining); "
            "Zero sets it flat to 0."
        )
        col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1, 1, 1.3])
        with col_m1:
            mask_enabled = st.checkbox("Enable mask", value=False, key="fep_mask_enabled")
        with col_m2:
            mask_min = st.number_input(
                "Mask min (°)", value=5.4, min_value=0.0, max_value=40.0, step=0.05,
                key="fep_mask_min", disabled=not mask_enabled,
            )
        with col_m3:
            mask_max = st.number_input(
                "Mask max (°)", value=5.7, min_value=0.0, max_value=40.0, step=0.05,
                key="fep_mask_max", disabled=not mask_enabled,
            )
        with col_m4:
            mask_method = st.radio(
                "Mask method", options=["interpolate", "zero"],
                horizontal=True, key="fep_mask_method", disabled=not mask_enabled,
            )

        if mask_enabled:
            # quick standalone preview of the mask region against the sample alone,
            # so you can see where it sits before running the full comparison
            try:
                x_prev, y_prev, _ = load_1d_pattern_any(
                    ab_sample, st.session_state.ai, st.session_state.mask,
                    st.session_state.npt, st.session_state.radial_min, st.session_state.radial_max,
                ) if ab_sample else (None, None, None)
                if x_prev is not None:
                    fig_prev = go.Figure()
                    fig_prev.add_trace(go.Scatter(x=x_prev, y=y_prev, mode="lines",
                                                   name="Sample (raw)",
                                                   line=dict(color="#2c3e50", width=1.2)))
                    fig_prev.add_vrect(x0=mask_min, x1=mask_max, fillcolor="#c0392b",
                                       opacity=0.15, layer="below", line_width=0,
                                       annotation_text="mask region", annotation_position="top left")
                    fig_prev.update_layout(
                        xaxis_title="2θ (°)", yaxis_title="Intensity",
                        xaxis=dict(range=[st.session_state.radial_min, st.session_state.radial_max]),
                        height=260, margin=dict(l=60, r=20, t=10, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_prev, use_container_width=True, key="fep_mask_preview")
            except Exception:
                pass  # preview is best-effort only; the real run below will surface any real error

        st.caption(
            "Sample, references, and the regions/mask above are shared by both sub-tabs "
            "below — only the combine step itself (and, for Divide, the floor) differs."
        )
        st.divider()

        ab_tab_sub, ab_tab_div = st.tabs(["➖  Subtract", "➗  Divide"])

        with ab_tab_sub:
            st.caption(
                "Physically motivated default — models FEP as an independent additive "
                "background."
            )
            render_fep_ab_comparison(
                method="subtract", key_prefix="ab_sub",
                ab_sample=ab_sample, ab_ref_a=ab_ref_a, ab_ref_b=ab_ref_b,
                ab_ref_a_label=ab_ref_a_label, ab_ref_b_label=ab_ref_b_label,
                ab_scale_min=ab_scale_min, ab_scale_max=ab_scale_max,
                ab_diag_min=ab_diag_min, ab_diag_max=ab_diag_max,
                ab_clip_negative=ab_clip_negative,
                mask_enabled=mask_enabled, mask_min=mask_min, mask_max=mask_max,
                mask_method=mask_method,
            )

        with ab_tab_div:
            st.caption(
                "⚠️ Test mode, not the recommended default — see the reasoning above. Can "
                "amplify noise where the reference is weak, and can suppress real Bragg "
                "intensity anywhere a sample peak overlaps a FEP feature (e.g. beta-glycine "
                "near the FEP ring) instead of removing background cleanly under it. Peak "
                "heights after division no longer represent raw photon counts."
            )
            ab_floor_frac = st.slider(
                "Reference floor (fraction of reference max)",
                min_value=0.005, max_value=0.20, value=0.02, step=0.005,
                key="ab_floor_frac",
                help=(
                    "The reference is clipped to at least this fraction of its own max before "
                    "dividing, to stop the ratio blowing up in low-count tails. Lower = closer "
                    "to a 'true' division but noisier; higher = more damped but also distorts "
                    "the correction more in those regions."
                ),
            )
            render_fep_ab_comparison(
                method="divide", key_prefix="ab_div",
                ab_sample=ab_sample, ab_ref_a=ab_ref_a, ab_ref_b=ab_ref_b,
                ab_ref_a_label=ab_ref_a_label, ab_ref_b_label=ab_ref_b_label,
                ab_scale_min=ab_scale_min, ab_scale_max=ab_scale_max,
                ab_diag_min=ab_diag_min, ab_diag_max=ab_diag_max,
                ab_clip_negative=ab_clip_negative,
                floor_frac=ab_floor_frac,
                mask_enabled=mask_enabled, mask_min=mask_min, mask_max=mask_max,
                mask_method=mask_method,
            )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # PRODUCTION — apply the decided method to every integrated frame
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### Apply to full batch")
        st.caption(
            "Once you've settled on a reference + method above, run it across every "
            "integrated frame from the Inline/Batch tabs, saving corrected patterns for "
            "the Baselining tab."
        )

        col_bm, col_bref = st.columns([1, 2])
        with col_bm:
            batch_method_choice = st.radio(
                "Method", options=["Subtract", "Divide"], horizontal=True,
                key="fep_batch_method_choice",
            )
            batch_method = batch_method_choice.lower()
        with col_bref:
            col1_, col2_ = st.columns([3, 1])
            with col1_:
                batch_ref_path = st.text_input(
                    "Reference (frame or .xy)",
                    value=st.session_state.get("_browse_fep_batch_ref", ""),
                    placeholder=r"E:/calib/Water_tri_ref.nxs  or  .xy",
                    key="fep_batch_ref_input",
                )
            with col2_:
                st.markdown("<br>", unsafe_allow_html=True)
                browse_button("Browse", "file", "_browse_fep_batch_ref", "Select batch reference")
            if st.session_state.get("_browse_fep_batch_ref"):
                batch_ref_path = st.session_state["_browse_fep_batch_ref"]

        default_in_dir = get_stage_dir("01_integrated") if st.session_state.data_dir else ""
        col_bi, col_bib = st.columns([3, 1])
        with col_bi:
            batch_in_dir = st.text_input(
                "Folder of integrated .xy files",
                value=st.session_state.get("_browse_fep_batch_in", default_in_dir),
                key="fep_batch_in_input",
            )
        with col_bib:
            st.markdown("<br>", unsafe_allow_html=True)
            browse_button("Browse", "folder", "_browse_fep_batch_in", "Select integrated folder")
        if st.session_state.get("_browse_fep_batch_in"):
            batch_in_dir = st.session_state["_browse_fep_batch_in"]

        if batch_method == "divide":
            batch_floor_frac = st.slider(
                "Reference floor (fraction of reference max)",
                min_value=0.005, max_value=0.20, value=0.02, step=0.005,
                key="fep_batch_floor_frac",
            )
        else:
            batch_floor_frac = 0.02

        st.caption(
            f"Uses the scale region ({ab_scale_min}–{ab_scale_max}°) and mask settings "
            f"above ({'enabled' if mask_enabled else 'disabled'})."
        )

        run_batch_fep = st.button(
            "Run FEP correction on full batch", type="primary",
            disabled=not (batch_ref_path and batch_in_dir and os.path.isdir(batch_in_dir)),
        )

        if run_batch_fep:
            in_files = sorted(glob.glob(os.path.join(batch_in_dir, "*_integrated.xy")))
            if not in_files:
                st.warning(f"No `*_integrated.xy` files found in {batch_in_dir}.")
            else:
                with st.spinner(f"Loading reference and correcting {len(in_files)} frames..."):
                    try:
                        ai, mask_arr, npt = st.session_state.ai, st.session_state.mask, st.session_state.npt
                        rmin, rmax = st.session_state.radial_min, st.session_state.radial_max
                        x_ref, y_ref, ref_kind = load_1d_pattern_any(
                            batch_ref_path, ai, mask_arr, npt, rmin, rmax
                        )

                        out_dir = get_stage_dir("02_fep_corrected")
                        progress = st.progress(0)
                        status_text = st.empty()
                        results = []

                        for i, fpath in enumerate(in_files):
                            stem = Path(fpath).stem.replace("_integrated", "")
                            status_text.text(f"{batch_method}ing {i+1}/{len(in_files)}: {stem}")
                            try:
                                data = np.loadtxt(fpath, skiprows=1, comments="#")
                                x_s, y_s = data[:, 0], data[:, 1]
                                _, _, y_comb, factor = combine_fep_1d(
                                    x_s, y_s, x_ref, y_ref, ab_scale_min, ab_scale_max,
                                    method=batch_method, clip_negative=ab_clip_negative,
                                    floor_frac=batch_floor_frac,
                                )
                                if mask_enabled:
                                    y_comb = mask_region(x_s, y_comb, mask_min, mask_max,
                                                          method=mask_method)
                                out_xy = os.path.join(out_dir, f"{stem}_fep{batch_method}.xy")
                                save_xy(out_xy, x_s, y_comb,
                                        header=f"2theta(deg) Intensity_FEP{batch_method}")
                                results.append({"file": stem, "status": "✓ OK",
                                                 "factor": f"{factor:.4f}"})
                            except Exception as e:
                                results.append({"file": stem, "status": f"✗ {e}", "factor": "—"})
                            progress.progress((i + 1) / len(in_files))

                        status_text.text("Complete.")
                        st.success(
                            f"FEP-corrected {len(in_files)} frames ({ref_kind} reference, "
                            f"{batch_method}) → {out_dir}"
                        )
                        import pandas as pd
                        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                        st.caption("Next: use the **Baselining** tab to MOR-baseline these frames "
                                   "and merge them into a final run pattern.")

                    except Exception as e:
                        st.error(f"Batch FEP correction failed: {e}")
                        st.exception(e)

    else:
        st.info("#### 2D FEP subtraction — pre-integration")
        st.markdown("""
This method will subtract the FEP background **in 2D, before azimuthal integration**.
This is significantly cleaner than 1D subtraction because:

- FEP polymer reflections are removed as **spots/arcs on the detector**, not as
  peaks in a 1D pattern where they overlap with sample Bragg peaks
- The integrated 1D pattern will contain only true sample diffraction
- Weak crystallisation signals will be more visible without the FEP hump

**Status:** in development — see `FEP_Subtraction/` folder (V0 and V1 trials).

**To connect your 2D method here when ready:**
Add your subtraction logic inside the clearly marked block in `app.py`:

```python
# ── 2D FEP SUBTRACTION — INSERT YOUR METHOD HERE ──────────────────────────
# Inputs available:
#   raw_frame   : np.ndarray, shape (2881, 2880), raw 2D detector frame
#   fep_frame   : np.ndarray, shape (2881, 2880), FEP reference 2D frame
#   scale       : float, scaling factor (calculated or user-supplied)
# Expected output:
#   subtracted_frame : np.ndarray, same shape as raw_frame
# ──────────────────────────────────────────────────────────────────────────
```

Once the 2D routine is working, the subtracted frame feeds straight into the
existing pyFAI integration and the rest of the pipeline is unchanged.
""")

        st.markdown("**Inputs that will be needed:**")
        col2d_a, col2d_b = st.columns(2)
        with col2d_a:
            st.text_input("Raw .nxs frame to process", placeholder=r"E:/run1/i11-1-122543.nxs",
                          disabled=True)
        with col2d_b:
            st.text_input("FEP reference .hdf or .nxs",
                          placeholder=r"E:/calib/FEP_reference.hdf", disabled=True)
        st.caption("These inputs are disabled until the 2D method is implemented.")


# ── TAB 4: BASELINING ─────────────────────────────────────────────────────────
with tab_baseline:
    st.markdown("### Baselining — per-frame MOR correction and merge")
    st.caption(
        "Applies morphological (MOR) baseline correction to individual frames, then "
        "combines and averages them into a single pattern for the whole run. Point this "
        "at either FEP-corrected frames (from the FEP subtraction tab) or raw integrated "
        "frames (from Inline/Batch) if you're skipping FEP correction."
    )

    fep_dir_guess = get_stage_dir("02_fep_corrected") if st.session_state.data_dir else ""
    int_dir_guess = get_stage_dir("01_integrated") if st.session_state.data_dir else ""
    default_bl_in = fep_dir_guess if (fep_dir_guess and glob.glob(os.path.join(fep_dir_guess, "*.xy"))) else int_dir_guess

    col_bi, col_bib = st.columns([3, 1])
    with col_bi:
        bl_in_dir = st.text_input(
            "Folder of patterns to baseline",
            value=st.session_state.get("_browse_bl_in", default_bl_in),
            key="bl_in_input",
            help="Defaults to the FEP-corrected folder if it has files, otherwise the "
                 "raw integrated folder.",
        )
    with col_bib:
        st.markdown("<br>", unsafe_allow_html=True)
        browse_button("Browse", "folder", "_browse_bl_in", "Select folder to baseline")
    if st.session_state.get("_browse_bl_in"):
        bl_in_dir = st.session_state["_browse_bl_in"]

    bl_files = sorted(glob.glob(os.path.join(bl_in_dir, "*.xy"))) if bl_in_dir and os.path.isdir(bl_in_dir) else []
    st.caption(f"{len(bl_files)} `.xy` file(s) found in this folder." if bl_in_dir else "")

    st.session_state.half_window = st.slider(
        "MOR baseline half_window", min_value=2, max_value=30,
        value=st.session_state.half_window,
        help="Same control as the sidebar (shared) — 5–10 typical for sharp PXRD peaks.",
    )

    # ── single-frame preview ────────────────────────────────────────────────
    if bl_files:
        st.markdown("##### Preview a single frame")
        preview_file = st.selectbox(
            "Frame", options=bl_files, format_func=os.path.basename, key="bl_preview_select",
        )
        if st.button("Preview baseline", key="bl_preview_btn"):
            try:
                data = np.loadtxt(preview_file, skiprows=1, comments="#")
                x_p, y_p = data[:, 0], data[:, 1]
                corrected_p, baseline_p = baseline_correct(x_p, y_p, st.session_state.half_window)
                fig_prev = go.Figure()
                fig_prev.add_trace(go.Scatter(x=x_p, y=y_p, mode="lines", name="Input",
                                               line=dict(color="#95a5a6", width=1)))
                fig_prev.add_trace(go.Scatter(x=x_p, y=baseline_p, mode="lines",
                                               name="MOR baseline",
                                               line=dict(color="#e74c3c", width=1.2, dash="dash")))
                fig_prev.add_trace(go.Scatter(x=x_p, y=corrected_p, mode="lines",
                                               name="Baselined",
                                               line=dict(color="#1a7a4a", width=1.5)))
                fig_prev.update_layout(
                    title=os.path.basename(preview_file),
                    xaxis_title="2θ (°)", yaxis_title="Intensity",
                    xaxis=dict(range=[st.session_state.radial_min, st.session_state.radial_max]),
                    hovermode="x unified", height=400,
                    margin=dict(l=60, r=20, t=40, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_prev, use_container_width=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")

    st.divider()

    # ── batch baseline ──────────────────────────────────────────────────────
    st.markdown("##### Batch baseline all frames in the folder")
    run_bl_batch = st.button(
        "Run batch baselining", type="primary",
        disabled=not bl_files,
    )
    if run_bl_batch:
        out_dir = get_stage_dir("03_baselined")
        progress = st.progress(0)
        status_text = st.empty()
        results = []
        for i, fpath in enumerate(bl_files):
            stem = Path(fpath).stem
            status_text.text(f"Baselining {i+1}/{len(bl_files)}: {stem}")
            try:
                data = np.loadtxt(fpath, skiprows=1, comments="#")
                x_b, y_b = data[:, 0], data[:, 1]
                corrected_b, _ = baseline_correct(x_b, y_b, st.session_state.half_window)
                out_xy = os.path.join(out_dir, f"{stem}_baselined.xy")
                save_xy(out_xy, x_b, corrected_b, header="2theta(deg) Intensity_baselined")
                results.append({"file": stem, "status": "✓ OK",
                                 "max_intensity": f"{corrected_b.max():.1f}"})
            except Exception as e:
                results.append({"file": stem, "status": f"✗ {e}", "max_intensity": "—"})
            progress.progress((i + 1) / len(bl_files))

        status_text.text("Complete.")
        st.success(f"Baselined {len(bl_files)} frames → {out_dir}")
        import pandas as pd
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    st.divider()

    # ── merge / average into final run pattern ──────────────────────────────
    st.markdown("### Combine → single pattern for the run")
    st.caption(
        "Averages all baselined frames onto a common 2θ grid into one representative "
        "pattern for the whole run. Run batch baselining above first."
    )
    baselined_glob = os.path.join(get_stage_dir("03_baselined"), "*_baselined.xy") if st.session_state.data_dir else ""
    if baselined_glob:
        render_merge_section(baselined_glob, "04_merged",
                              title="Merge all baselined frames → final PXRD pattern")
    else:
        st.info("Set your data directory in the sidebar first.")


# ── TAB 6: STATUS ─────────────────────────────────────────────────────────────
with tab_status:
    st.markdown("### Session status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Calibration**")
        if st.session_state.calib_loaded:
            st.markdown('<p class="status-ok">● Loaded</p>', unsafe_allow_html=True)
            st.code(f"PONI:  {st.session_state.poni_path}\nMask:  {st.session_state.mask_path}")
            if st.session_state.ai is not None:
                ai = st.session_state.ai
                try:
                    st.markdown(f"""
| Parameter | Value |
|---|---|
| Distance (m) | `{ai.dist:.4f}` |
| poni1 (m) | `{ai.poni1:.4f}` |
| poni2 (m) | `{ai.poni2:.4f}` |
| Wavelength (Å) | `{ai.wavelength*1e10:.5f}` |
""")
                except Exception:
                    pass
        else:
            st.markdown('<p class="status-warn">● Not loaded</p>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Data session**")
        if st.session_state.data_dir:
            nxs_files = get_nxs_files(st.session_state.data_dir)
            stage_root = st.session_state.global_output_dir or st.session_state.data_dir
            st.markdown('<p class="status-ok">● Directory set</p>', unsafe_allow_html=True)
            st.code(
                f"Data:    {st.session_state.data_dir}\n"
                f"Run:     {st.session_state.run_name}\n"
                f"Stages:  {stage_root}/{st.session_state.run_name or 'run'}_<stage>/\n"
                f"Frames:  {len(nxs_files)} found\n"
                f"Done:    {len(st.session_state.processed_files)} integrated this session"
            )
        else:
            st.markdown('<p class="status-warn">● Not set</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Processing parameters**")
    st.code(
        f"half_window : {st.session_state.half_window}\n"
        f"2θ range    : {st.session_state.radial_min}° – {st.session_state.radial_max}°\n"
        f"npt         : {st.session_state.npt}"
    )

    st.divider()
    st.markdown("**Pipeline stages this run**")
    if st.session_state.data_dir:
        for stage in ["01_integrated", "02_fep_corrected", "03_baselined", "04_merged"]:
            d = get_stage_dir(stage)
            n = len(glob.glob(os.path.join(d, "*.xy")))
            icon = "✓" if n > 0 else "○"
            st.caption(f"{icon} `{stage}` — {n} file(s) — `{d}`")
    else:
        st.caption("Set a data directory in the sidebar to see stage folders.")

    st.divider()
    st.markdown("**Pipeline roadmap**")
    st.markdown("""
| Phase | Status | Description |
|---|---|---|
| Calibration | ✅ Done | pyFAI .poni + mask, verified once per beamtime |
| Inline / Batch integration | ✅ Done | Raw frame → integrated 1D pattern only |
| FEP subtraction | ✅ Done | Subtract/divide reference comparison + intense-peak mask + full-batch run |
| Baselining | ✅ Done | Per-frame MOR baseline + merge/average into final run pattern |
| Pattern viewer | ✅ Done | Compare patterns from any stage folder |
| ML classifier | 🔜 Next | Train random forest on v7 triage labels |
| Inline ML | 🔜 Future | Auto-classify frames; skip baselining backgrounds |
| 2D FEP subtraction | 🔜 Future | Subtract FEP in 2D before integration |
""")
    st.caption("Tabs to add later: 00 raw sort · 02 manual triage · 06 FEP subtraction · 07 stacking")
