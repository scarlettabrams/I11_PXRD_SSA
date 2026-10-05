#Launching the app
# cd C:\Users\ezxsa27\I11_PXRD_SSA
# conda activate I11_PXRD_SSA_env
# streamlit run app.py

"""
I11 PXRD Processing Pipeline
Streamlit app — Phase 1: inline 04A core loop
Run with: streamlit run app.py
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


# ── tkinter file/folder picker ─────────────────────────────────────────────────
def pick_path(mode="folder", title="Select", filetypes=None, key=None):
    """
    Opens a native Windows file/folder dialog via tkinter and stores the
    result in st.session_state[key]. Returns the selected path or None.

    mode : "folder" | "file"
    key  : session_state key to write the result into
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()          # hide the empty tkinter window
    root.wm_attributes("-topmost", True)   # bring dialog to front

    if mode == "folder":
        path = filedialog.askdirectory(title=title)
    else:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes or [("All files", "*.*")],
        )

    root.destroy()

    if path and key:
        st.session_state[key] = path
    return path or None


def browse_button(label, mode, session_key, title="Select", filetypes=None):
    """Render a compact browse button that writes to session_state[session_key]."""
    if st.button(f"📂 {label}", key=f"browse_{session_key}",
                 use_container_width=True):
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
    col_dd, col_db = st.columns([3, 1])
    with col_dd:
        data_dir_input = st.text_input(
            "Directory containing .nxs files",
            value=st.session_state.get("_browse_data_dir", st.session_state.data_dir),
            placeholder=r"E:/beamtime/RAW",
            help="Flat directory where frames arrive during collection",
            key="_data_dir_text",
        )
    with col_db:
        st.markdown("<br>", unsafe_allow_html=True)
        browse_button("Browse", "folder", "_browse_data_dir", "Select data directory")
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
            # create output folder alongside the data directory
            out = os.path.join(data_dir_input, f"04A_{run_name_input.strip()}_baselined")
            os.makedirs(out, exist_ok=True)
            st.session_state.output_dir = out
            st.session_state.processed_files = []
            nxs = get_nxs_files(data_dir_input)
            st.success(f"Found {len(nxs)} .nxs files. Output → {out}")

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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — tabs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🎯 I11 PXRD inline pipeline")
st.caption("Diamond Light Source · Beamline I11 · Flow crystallisation")

tab_calib, tab_inline, tab_batch, tab_viewer, tab_fep, tab_status = st.tabs([
    "🎯  Calibration",
    "▶  Inline loop",
    "⚡  Batch process",
    "📊  Pattern viewer",
    "🧪  FEP subtraction",
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
    st.markdown("#### Step 2 — Run pyFAI calibration")
    st.caption(
        "Click the button below to launch the pyFAI calibration GUI in a new window. "
        "In pyFAI: set wavelength (~0.826 Å for I11), select CeO2 calibrant, "
        "apply your detector mask, pick rings, fit geometry, then **save the .poni file** "
        "and **export the mask as .npy** before closing."
    )

    col_pyfai, col_info = st.columns([1, 2])
    with col_pyfai:
        launch_pyfai = st.button(
            "Launch pyFAI calibration GUI →",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.calib_hdf_path),
            help="Load a calibration image in Step 1 first.",
        )
    with col_info:
        st.info(
            "pyFAI will open in a **separate window**. "
            "Complete calibration there, save your files, then come back here for Step 3."
        )

    if launch_pyfai:
        import subprocess, sys
        hdf = st.session_state.calib_hdf_path

        # Find pyFAI-calib2.exe — it lives in Scripts/ next to python.exe
        # Note: Windows is case-insensitive but the actual filename is pyFAI-calib2.exe
        python_dir = Path(sys.executable).parent
        scripts_dir = python_dir / "Scripts"
        possible_exes = [
            scripts_dir / "pyFAI-calib2.exe",
            scripts_dir / "pyfai-calib2.exe",   # fallback lowercase
            python_dir  / "pyFAI-calib2.exe",
        ]
        exe = next((str(p) for p in possible_exes if p.exists()), None)

        if exe:
            cmd = [exe, hdf]
        else:
            # last resort: module invocation
            cmd = [sys.executable, "-m", "pyFAI.app.calib2", hdf]

        # always show the manual command as a guaranteed fallback
        st.markdown("**Command to run manually if needed:**")
        st.code(" ".join(f'"{c}"' if " " in c else c for c in cmd),
                language="bash")
        st.caption(
            "Copy and paste this into your terminal (with I11_PXRD_SSA_env active) "
            "if pyFAI does not open automatically."
        )

        try:
            if os.name == "nt":
                DETACHED_PROCESS         = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                subprocess.Popen(
                    cmd,
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            st.success(
                "pyFAI-calib2 launch requested — it can take 15–30 seconds to open. "
                "If nothing appears after 30 seconds, use the manual command above."
            )
        except Exception as e:
            st.error(f"Auto-launch failed: {e} — use the manual command above instead.")

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
        "Process frames as they arrive. Step through with Prev / Next, "
        "or pick from the dropdown. Use during active data collection."
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
                "Process this frame →",
                type="primary",
                use_container_width=False,
            )

            if process_btn:
                with st.spinner(f"Processing {frame_name}..."):
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
                        corrected, baseline = baseline_correct(
                            x, y, st.session_state.half_window
                        )
                        hw = st.session_state.half_window
                        out_stem = f"{frame_name}_hw{hw}_baselined"
                        xy_path = os.path.join(
                            st.session_state.output_dir, out_stem + ".xy"
                        )
                        save_xy(xy_path, x, corrected)

                        st.session_state.last_pattern = (x, y, baseline, corrected)
                        st.session_state.last_pattern_name = frame_name
                        if selected_file not in st.session_state.processed_files:
                            st.session_state.processed_files.append(selected_file)

                        st.success(f"✓ Saved: {os.path.basename(xy_path)}")

                        # interactive plotly chart
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(
                            x=x, y=y, mode="lines", name="Raw integrated",
                            line=dict(color="#95a5a6", width=1),
                        ))
                        fig_p.add_trace(go.Scatter(
                            x=x, y=baseline, mode="lines", name="Baseline (MOR)",
                            line=dict(color="#e74c3c", width=1.2, dash="dash"),
                        ))
                        fig_p.add_trace(go.Scatter(
                            x=x, y=corrected, mode="lines", name="Corrected",
                            line=dict(color="#1a7a4a", width=1.5),
                        ))
                        fig_p.update_layout(
                            title=f"{frame_name} — baseline corrected",
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
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Max intensity", f"{corrected.max():.1f}")
                        c2.metric("Peak 2θ (°)", f"{x[np.argmax(corrected)]:.3f}")
                        c3.metric("half_window", hw)
                        c4.metric("Frames processed", len(st.session_state.processed_files))

                        # auto-advance hint
                        if st.session_state.frame_index < n_frames - 1:
                            st.info("Click **Next ▶** to move to the next frame.")

                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                        st.exception(e)


# ── TAB 2: BATCH PROCESS ─────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch processing — all frames in directory")
    st.caption(
        "Process every unprocessed .nxs frame in the data directory at once. "
        "Use this at the end of a run, or to catch up after a gap."
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

        if st.button("Run batch processing", type="primary"):
            target = to_process if skip_processed else nxs_files
            if not target:
                st.info("Nothing to process.")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                results = []

                for i, filepath in enumerate(target):
                    frame_name = Path(filepath).stem
                    status_text.text(f"Processing {i+1}/{len(target)}: {frame_name}")
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
                        corrected, baseline = baseline_correct(
                            x, y, st.session_state.half_window
                        )
                        hw = st.session_state.half_window
                        out_stem = f"{frame_name}_hw{hw}_baselined"
                        xy_path = os.path.join(
                            st.session_state.output_dir, out_stem + ".xy"
                        )
                        save_xy(xy_path, x, corrected)
                        results.append({"file": frame_name, "status": "✓ OK",
                                        "max_intensity": f"{corrected.max():.1f}"})
                        if filepath not in st.session_state.processed_files:
                            st.session_state.processed_files.append(filepath)
                    except Exception as e:
                        results.append({"file": frame_name, "status": f"✗ {e}",
                                        "max_intensity": "—"})

                    progress.progress((i + 1) / len(target))

                status_text.text("Complete.")
                st.success(f"Processed {len(target)} frames → {st.session_state.output_dir}")

                import pandas as pd
                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── 04B MERGE SECTION ──────────────────────────────────────────────
        st.divider()
        st.markdown("### Merge baselined frames → single PXRD pattern (04B)")
        st.caption(
            "Averages all baselined .xy files in the output folder into one "
            "representative pattern for this run. Run this after batch processing is complete."
        )

        xy_in_output = sorted(glob.glob(
            os.path.join(st.session_state.output_dir, "*_baselined.xy")
        )) if st.session_state.output_dir else []

        if not xy_in_output:
            st.info("No baselined .xy files found yet — run batch processing above first.")
        else:
            st.markdown(f"**{len(xy_in_output)}** baselined frames ready to merge.")

            if st.button("Merge all frames → final PXRD pattern", type="primary"):
                with st.spinner("Merging..."):
                    try:
                        intensities = []
                        two_theta_ref = None

                        for fpath in xy_in_output:
                            data = np.loadtxt(fpath, skiprows=1, comments="#")
                            if data.ndim < 2 or data.shape[1] < 2:
                                continue
                            x_i = data[:, 0]
                            y_i = data[:, 1]
                            # interpolate all patterns onto the first file's 2θ grid
                            if two_theta_ref is None:
                                two_theta_ref = x_i
                            y_interp = np.interp(two_theta_ref, x_i, y_i)
                            intensities.append(y_interp)

                        if not intensities:
                            st.error("No valid .xy files could be read.")
                        else:
                            merged = np.mean(intensities, axis=0)
                            std_pattern = np.std(intensities, axis=0)

                            # save merged .xy
                            run = st.session_state.run_name
                            hw = st.session_state.half_window
                            merged_stem = f"{run}_hw{hw}_merged_final"
                            merged_dir = os.path.join(
                                st.session_state.data_dir,
                                f"04B_{run}_merged"
                            )
                            os.makedirs(merged_dir, exist_ok=True)
                            xy_out = os.path.join(merged_dir, merged_stem + ".xy")
                            png_out = os.path.join(merged_dir, merged_stem + ".png")

                            save_xy(xy_out, two_theta_ref, merged,
                                    header="2theta(deg) Intensity(averaged)")

                            # plot merged pattern
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(two_theta_ref, merged, color="#1a7a4a",
                                    linewidth=1.0, label=f"Merged ({len(intensities)} frames)")
                            ax.fill_between(
                                two_theta_ref,
                                merged - std_pattern,
                                merged + std_pattern,
                                color="#1a7a4a", alpha=0.15, label="±1σ"
                            )
                            ax.set_xlabel("2θ (°)", fontsize=12)
                            ax.set_ylabel("Intensity (a.u.)", fontsize=12)
                            ax.set_title(
                                f"{run} — merged PXRD pattern "
                                f"({len(intensities)} frames, half_window={hw})",
                                fontsize=11
                            )
                            ax.set_xlim(st.session_state.radial_min,
                                        st.session_state.radial_max)
                            ax.legend(fontsize=10)
                            ax.grid(alpha=0.2)
                            fig.tight_layout()
                            fig.savefig(png_out, dpi=300, bbox_inches="tight")
                            st.pyplot(fig)
                            plt.close(fig)

                            st.success(
                                f"✓ Merged {len(intensities)} frames\n\n"
                                f"Saved: `{xy_out}`"
                            )

                            # summary metrics
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Frames merged", len(intensities))
                            c2.metric("Max intensity", f"{merged.max():.1f}")
                            c3.metric("Peak 2θ (°)",
                                      f"{two_theta_ref[np.argmax(merged)]:.3f}")

                    except Exception as e:
                        st.error(f"Merge failed: {e}")
                        st.exception(e)


# ── TAB 3: PATTERN VIEWER ─────────────────────────────────────────────────────
with tab_viewer:
    st.markdown("### Pattern viewer — interactive comparison")
    st.caption(
        "Zoom, pan, and hover over patterns. Click legend entries to toggle traces. "
        "Select patterns below to overlay them."
    )

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        xy_dir = st.text_input(
            "Folder containing .xy files",
            value=st.session_state.get("_browse_xy_dir",
                  st.session_state.output_dir or st.session_state.data_dir),
            placeholder=r"E:/beamtime/RAW/04A_Run1_baselined",
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
        "Remove FEP tubing scattering contributions from your PXRD patterns. "
        "Two methods available — 1D post-integration (current) and 2D pre-integration (future)."
    )

    # ── method toggle ──────────────────────────────────────────────────────────
    fep_method = st.radio(
        "Subtraction method",
        options=["1D — post-integration (available)", "2D — pre-integration (coming soon)"],
        horizontal=True,
        help=(
            "1D: subtract a scaled FEP reference pattern from your merged 1D .xy file. "
            "2D: subtract FEP in 2D before azimuthal integration — cleaner but requires "
            "a working 2D FEP subtraction routine."
        ),
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 1D METHOD
    # ══════════════════════════════════════════════════════════════════════════
    if fep_method.startswith("1D"):
        st.markdown("#### 1D FEP subtraction — post-integration")
        st.caption(
            "Loads a merged/baselined sample .xy and a representative FEP background .xy "
            "collected under the same geometry. Scales the FEP pattern to match the sample "
            "background region, subtracts, and saves the cleaned pattern."
        )

        col_s, col_f = st.columns(2)
        with col_s:
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                sample_xy = st.text_input(
                    "Sample .xy file (merged, baselined)",
                    value=st.session_state.get("_browse_sample_xy", ""),
                    placeholder=r"E:/run1/04B_merged/Run1_merged_final.xy",
                    help="Output from 04B merge step or 05 post-merge baselining.",
                )
            with col_s2:
                st.markdown("<br>", unsafe_allow_html=True)
                browse_button("Browse", "file", "_browse_sample_xy", "Select sample .xy",
                              filetypes=[("XY files", "*.xy"), ("All files", "*.*")])
            if st.session_state.get("_browse_sample_xy"):
                sample_xy = st.session_state["_browse_sample_xy"]

        with col_f:
            col_f1, col_f2 = st.columns([3, 1])
            with col_f1:
                fep_xy = st.text_input(
                    "FEP reference .xy file",
                    value=st.session_state.get("_browse_fep_xy", ""),
                    placeholder=r"E:/calib/FEP_background.xy",
                    help="Integrated 1D pattern from an empty FEP tube collected at same geometry.",
                )
            with col_f2:
                st.markdown("<br>", unsafe_allow_html=True)
                browse_button("Browse", "file", "_browse_fep_xy", "Select FEP reference .xy",
                              filetypes=[("XY files", "*.xy"), ("All files", "*.*")])
            if st.session_state.get("_browse_fep_xy"):
                fep_xy = st.session_state["_browse_fep_xy"]

        st.markdown("**Scaling region** — 2θ range used to match FEP intensity to sample")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            scale_min = st.number_input(
                "Scale region min (°)", value=2.0, min_value=0.0, max_value=10.0, step=0.5,
                help="Pick a region where only FEP scattering is present, no sample peaks.",
            )
        with col_r2:
            scale_max = st.number_input(
                "Scale region max (°)", value=4.0, min_value=0.0, max_value=15.0, step=0.5,
            )

        clip_negative = st.checkbox(
            "Clip negative values to zero after subtraction", value=True,
            help="Removes artefacts where over-subtraction produces negative intensity.",
        )

        run_fep_1d = st.button("Run 1D FEP subtraction", type="primary",
                               disabled=not (sample_xy and fep_xy))

        if run_fep_1d:
            if not os.path.exists(sample_xy):
                st.error(f"Sample file not found:\n{sample_xy}")
            elif not os.path.exists(fep_xy):
                st.error(f"FEP reference file not found:\n{fep_xy}")
            else:
                with st.spinner("Subtracting FEP..."):
                    try:
                        # load both patterns
                        s_data = np.loadtxt(sample_xy,  skiprows=1, comments="#")
                        f_data = np.loadtxt(fep_xy,     skiprows=1, comments="#")
                        x_s, y_s = s_data[:, 0], s_data[:, 1]
                        x_f, y_f = f_data[:, 0], f_data[:, 1]

                        # interpolate FEP onto sample 2θ grid
                        y_f_interp = np.interp(x_s, x_f, y_f)

                        # calculate scale factor from the user-defined region
                        mask_region = (x_s >= scale_min) & (x_s <= scale_max)
                        if mask_region.sum() < 3:
                            st.error(
                                "Scaling region contains fewer than 3 points — "
                                "widen the range or check your 2θ limits."
                            )
                        else:
                            s_region = y_s[mask_region]
                            f_region = y_f_interp[mask_region]
                            # scale factor: ratio of mean intensities in the region
                            scale_factor = (
                                np.mean(s_region) / np.mean(f_region)
                                if np.mean(f_region) > 0 else 1.0
                            )
                            y_f_scaled = y_f_interp * scale_factor
                            y_subtracted = y_s - y_f_scaled
                            if clip_negative:
                                y_subtracted = np.clip(y_subtracted, 0, None)

                            # save output
                            sample_stem = Path(sample_xy).stem
                            out_dir = os.path.join(
                                Path(sample_xy).parent,
                                "06_fep_subtracted"
                            )
                            os.makedirs(out_dir, exist_ok=True)
                            out_xy  = os.path.join(out_dir, f"{sample_stem}_FEPsub.xy")
                            out_png = os.path.join(out_dir, f"{sample_stem}_FEPsub.png")
                            save_xy(out_xy, x_s, y_subtracted,
                                    header="2theta(deg) Intensity_FEPsubtracted")

                            # interactive plot
                            fig_fep = go.Figure()
                            fig_fep.add_trace(go.Scatter(
                                x=x_s, y=y_s, mode="lines", name="Sample (input)",
                                line=dict(color="#95a5a6", width=1),
                            ))
                            fig_fep.add_trace(go.Scatter(
                                x=x_s, y=y_f_scaled, mode="lines",
                                name=f"FEP scaled (×{scale_factor:.3f})",
                                line=dict(color="#e74c3c", width=1, dash="dash"),
                            ))
                            fig_fep.add_trace(go.Scatter(
                                x=x_s, y=y_subtracted, mode="lines",
                                name="FEP subtracted",
                                line=dict(color="#1a7a4a", width=1.5),
                            ))
                            # mark scaling region
                            fig_fep.add_vrect(
                                x0=scale_min, x1=scale_max,
                                fillcolor="#f39c12", opacity=0.10,
                                layer="below", line_width=0,
                                annotation_text="scale region",
                                annotation_position="top left",
                                annotation_font_size=10,
                            )
                            fig_fep.update_layout(
                                title=f"1D FEP subtraction — scale factor {scale_factor:.4f}",
                                xaxis_title="2θ (°)",
                                yaxis_title="Intensity",
                                xaxis=dict(
                                    tickfont=dict(color="black"),
                                    title_font=dict(color="black"),
                                    linecolor="black", gridcolor="#e0e0e0",
                                ),
                                yaxis=dict(
                                    tickfont=dict(color="black"),
                                    title_font=dict(color="black"),
                                    linecolor="black", gridcolor="#e0e0e0",
                                ),
                                font=dict(color="black"),
                                paper_bgcolor="white",
                                plot_bgcolor="white",
                                hovermode="x unified",
                                legend=dict(font=dict(color="black")),
                                height=450,
                                margin=dict(l=60, r=20, t=60, b=50),
                            )
                            st.plotly_chart(fig_fep, use_container_width=True)

                            # save static png
                            import matplotlib
                            fig_s, ax_s = plt.subplots(figsize=(10, 4))
                            ax_s.plot(x_s, y_s, color="#95a5a6", lw=0.8,
                                      label="Sample (input)")
                            ax_s.plot(x_s, y_f_scaled, color="#e74c3c", lw=1,
                                      ls="--", label=f"FEP scaled (×{scale_factor:.3f})")
                            ax_s.plot(x_s, y_subtracted, color="#1a7a4a", lw=1.2,
                                      label="FEP subtracted")
                            ax_s.set_xlabel("2θ (°)"); ax_s.set_ylabel("Intensity")
                            ax_s.legend(fontsize=8); ax_s.grid(alpha=0.2)
                            fig_s.tight_layout()
                            fig_s.savefig(out_png, dpi=300, bbox_inches="tight")
                            plt.close(fig_s)

                            st.success(
                                f"✓ Scale factor: {scale_factor:.4f}\n\n"
                                f"Saved: `{out_xy}`"
                            )
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Scale factor", f"{scale_factor:.4f}")
                            c2.metric("Max intensity (subtracted)",
                                      f"{y_subtracted.max():.1f}")
                            c3.metric("Peak 2θ (°)",
                                      f"{x_s[np.argmax(y_subtracted)]:.3f}")
                            st.caption(
                                "If the subtraction looks over- or under-scaled, adjust "
                                "the scaling region to a cleaner FEP-only 2θ window and rerun."
                            )

                    except Exception as e:
                        st.error(f"Subtraction failed: {e}")
                        st.exception(e)

    # ══════════════════════════════════════════════════════════════════════════
    # 2D METHOD — placeholder, ready to be filled when method is developed
    # ══════════════════════════════════════════════════════════════════════════
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
            st.markdown('<p class="status-ok">● Directory set</p>', unsafe_allow_html=True)
            st.code(
                f"Data:    {st.session_state.data_dir}\n"
                f"Run:     {st.session_state.run_name}\n"
                f"Output:  {st.session_state.output_dir}\n"
                f"Frames:  {len(nxs_files)} found\n"
                f"Done:    {len(st.session_state.processed_files)} processed"
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
    st.markdown("**Pipeline roadmap**")
    st.markdown("""
| Phase | Status | Description |
|---|---|---|
| Phase 1 — Streamlit app | ✅ In progress | Calibration · inline 04A loop · batch · 04B merge · 1D FEP sub · viewer |
| Phase 2 — ML classifier | 🔜 Next | Train random forest on v7 triage labels |
| Phase 3 — Inline ML | 🔜 Future | Auto-classify frames; skip baselining backgrounds |
| 2D FEP subtraction | 🔜 Future | Subtract FEP in 2D before integration; plug into FEP tab |
""")
    st.caption("Tabs to add later: 00 raw sort · 02 manual triage · 06 FEP subtraction · 07 stacking")
