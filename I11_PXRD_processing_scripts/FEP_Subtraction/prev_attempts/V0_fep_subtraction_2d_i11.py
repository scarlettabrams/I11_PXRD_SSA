"""
fep_subtraction_2d_i11.py
=============================================================
2D frame-level FEP background subtraction for Diamond Light Source
beamline I11, Pixium detector, tri-segmented flow PXRD experiments.

EXPERIMENT CONTEXT
------------------
Crystallising solution boluses flow through FEP tubing in a tri-segmented
stream (solution bolus | N2 gas | Galden carrier fluid) at ~5.5 ml/min.
A laser-diode trigger attempts to align each ~10 s X-ray snapshot with the
crystal-containing region of a bolus.  Flow instabilities mean many frames
miss the crystal and instead contain gas, Galden, or unsaturated solution.

PIPELINE POSITION
-----------------
  Raw .nxs files arrive from I11 EH2
        |
        v
  [FRAME CLASSIFIER]   <- score every frame; export ML-ready CSV
        |
        +-- crystal frames  -->  [SUBTRACTOR]  -->  clean 2D frame
        |                              ^
        +-- background frames  -->  [BACKGROUND BUILDER]
                                      (averaged, best-available)
        |
        v
  pyFAI integrate1d with existing .poni calibration  (unchanged)
        |
        v
  Clean 1D pattern


FRAME CLASSIFICATION STRATEGY
------------------------------
Because FEP scattering currently overwhelms automated Bragg detection,
the classifier uses proxy metrics computable from the raw frame alone:

  total_counts       - total pixel sum
  spatial_variance   - variance across pixels; Bragg spots raise this
  ring_score         - azimuthal variance in FEP ring annulus;
                       crystal spots break ring symmetry -> higher score
  hotspot_ratio      - fraction of pixels above 5x median;
                       Bragg spots create localised hotspots
  radial_contrast    - p90/p50 of radial profile; peaked = rings/spots

These metrics populate a CSV with a blank human_label column.
As you label frames visually, the CSV becomes your ML training dataset.

BACKGROUND TYPE CLASSIFICATION
-------------------------------
Non-crystal frames are sub-classified:
  gas frame      - very low total counts (beam through N2 gap)
  solution frame - moderate counts (Galden + solution, no Bragg)

Solution frames are preferred as background: they match the scattering
environment during a crystal hit more closely than gas frames.

SCALE FACTOR
------------
Auto-computed from NeXus count_time metadata.
Override with scale_factor=float when needed.

DEPENDENCIES
------------
    pip install numpy h5py

Optional:
    pip install matplotlib   (diagnostic plots)
    pip install pyFAI        (downstream integration, unchanged from current workflow)

Author : SSA
Date   : 2026-04
"""

from __future__ import annotations

import csv
import datetime
import logging
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# I11 / Pixium NeXus path constants
# Mirrors load_pixium_frame_hdf5() known_paths in your existing notebook
# =============================================================================

I11_DETECTOR_PATHS: tuple[str, ...] = (
    "entry1/pixium_hdf/data",
    "entry/data/data",
    "entry1/data/data",
    "entry1/Pixium10:detector/data",
    "entry/Pixium10:detector/data",
    "entry1/detector/data",
    "entry/detector/data",
)

I11_EXPOSURE_PATHS: tuple[str, ...] = (
    "entry1/instrument/detector/count_time",
    "entry1/instrument/detector/exposure_time",
    "entry/instrument/detector/count_time",
    "entry/instrument/detector/exposure_time",
    "entry1/count_time",
    "entry/count_time",
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FrameMetrics:
    """
    Per-frame metrics for classification and ML training export.

    All numeric fields are computed from the raw 2D pixel array with no
    calibration file required.  Fill in human_label during manual review
    to build your ML training dataset.

    human_label values (suggested):
        "crystal"   - clear Bragg diffraction
        "solution"  - solution + Galden, no Bragg
        "gas"       - N2 gap, very low counts
        "mixed"     - ambiguous phase boundary hit
        "reject"    - artefact / bad frame
    """
    # Identification
    filepath:           str
    frame_index:        int
    collection_number:  str

    # Raw metrics
    total_counts:       float
    mean_counts:        float
    spatial_variance:   float
    hotspot_ratio:      float
    radial_contrast:    float
    ring_score:         float
    low_angle_power:    float

    # Derived
    crystal_score:      float   # 0-1 heuristic; replaced by ML model later

    # Classification
    auto_class:         str     # "crystal" | "solution" | "gas" | "uncertain"
    human_label:        str = ""
    notes:              str = ""


@dataclass
class SubtractionResult:
    """Output bundle from a single frame subtraction."""
    corrected_frame:     np.ndarray
    scale_factor:        float
    background_type:     str
    n_background_frames: int
    data_filepath:       str
    output_filepath:     str
    frame_metrics:       FrameMetrics


@dataclass
class FrameMetricConfig:
    """Runtime-tunable parameters for robust frame scoring."""
    beam_excl_px: int = 60
    outer_excl_px: Optional[int] = 1200
    sigma_multiplier: float = 5.0
    use_mad_noise: bool = True
    detect_fep_ring: bool = True
    fixed_fep_ring_r: Optional[int] = None
    fep_search_min_px: int = 80
    fep_search_max_px: int = 600
    smooth_window: int = 11
    crystal_threshold: float = 0.22 / 100.0
    solution_threshold: float = 0.05 / 100.0
    load_retries: int = 2
    load_retry_sleep_s: float = 0.2


@dataclass
class FrameEvalStatus:
    """Structured per-frame status record for robust batch runs."""
    filepath: str
    frame_index: int
    status: str
    error: str = ""
    auto_class: str = "uncertain"
    crystal_score: float = 0.0
    spot_density: float = 0.0
    noise_sigma: float = 0.0
    spot_threshold: float = 0.0
    fep_ring_r: int = -1


# =============================================================================
# I/O helpers
# =============================================================================

def _find_detector_path(
    f: h5py.File,
    filename: str,
    override: Optional[str],
) -> str:
    if override is not None:
        if override not in f:
            raise KeyError(f"detector_path '{override}' not found in {filename}.")
        return override
    for p in I11_DETECTOR_PATHS:
        if p in f:
            logger.debug("Detector dataset: %s", p)
            return p
    # last resort: largest 2D/3D dataset by pixel count
    found: list[str] = []
    f.visititems(
        lambda name, obj: found.append(name)
        if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2
        else None
    )
    if not found:
        raise RuntimeError(
            f"No 2D dataset found in {filename}. Pass detector_path explicitly."
        )
    best = max(found, key=lambda n: int(np.prod(f[n].shape[-2:])))
    logger.warning("Falling back to largest 2D dataset: %s", best)
    return best


def _load_pixium_frame(
    filepath: Union[str, Path],
    frame_indices: Optional[Union[int, Sequence[int]]] = None,
    detector_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load Pixium frame(s) from a Diamond I11 NeXus file.

    Returns float64 array shape (rows, cols) or (N, rows, cols).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        ds_path = _find_detector_path(f, filepath.name, detector_path)
        ds = f[ds_path]
        shape = ds.shape

        if len(shape) == 2:
            return ds[()].astype(np.float64)

        if len(shape) == 3:
            n = shape[0]
            if frame_indices is None:
                return ds[()].astype(np.float64)
            if isinstance(frame_indices, (int, np.integer)):
                idx = int(frame_indices)
                if not 0 <= idx < n:
                    raise IndexError(
                        f"frame_index {idx} out of range "
                        f"({filepath.name} has {n} frames)."
                    )
                return ds[idx].astype(np.float64)
            indices = list(frame_indices)
            bad = [i for i in indices if not 0 <= i < n]
            if bad:
                raise IndexError(
                    f"frame_indices {bad} out of range "
                    f"({filepath.name} has {n} frames)."
                )
            return ds[indices].astype(np.float64)

        raise ValueError(
            f"Dataset '{ds_path}' has unexpected shape {shape}. "
            "Expected 2D or 3D array."
        )


def _read_exposure_time(filepath: Union[str, Path]) -> Optional[float]:
    """Read exposure time in seconds from NeXus metadata."""
    with h5py.File(filepath, "r") as f:
        for p in I11_EXPOSURE_PATHS:
            if p in f:
                return float(np.squeeze(f[p][()]))
    return None


def _read_collection_number(filepath: Path) -> str:
    """Extract I11 collection number from filename stem."""
    stem = filepath.stem
    # i11-1-NNNNNN
    for part in reversed(stem.split("-")):
        if part.isdigit():
            return part
    # pixium_NNNNNN
    if "_" in stem:
        tail = stem.split("_")[-1]
        if tail.isdigit():
            return tail
    return stem


# =============================================================================
# Frame metrics and classification
# =============================================================================

def _smooth_1d(x: np.ndarray, window: int = 11) -> np.ndarray:
    """Simple edge-preserving moving-average smoothing for radial profiles."""
    w = max(1, int(window))
    if w == 1:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64) / float(w)
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _mad_sigma(x: np.ndarray) -> float:
    """Robust sigma estimator based on MAD."""
    if x.size == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-12)


def _safe_load_pixium_frame(
    filepath: Union[str, Path],
    frame_index: int = 0,
    detector_path: Optional[str] = None,
    retries: int = 2,
    sleep_s: float = 0.2,
) -> np.ndarray:
    """Load a frame with retry/backoff for transient HDF5 I/O issues."""
    last_exc: Optional[Exception] = None
    for _ in range(max(0, retries) + 1):
        try:
            frame = _load_pixium_frame(filepath, frame_indices=frame_index, detector_path=detector_path)
            if frame.ndim == 3:
                frame = frame[0]
            if frame.ndim != 2:
                raise ValueError(f"Expected 2D frame, got shape {frame.shape}")
            if not np.isfinite(frame).all():
                raise ValueError("Frame contains NaN/Inf values")
            return frame.astype(np.float64)
        except Exception as exc:          # noqa: BLE001
            last_exc = exc
            time.sleep(max(0.0, float(sleep_s)))
    raise RuntimeError(f"Failed to load frame from {filepath}") from last_exc


def _auto_detect_fep_ring_r(
    radial_profile: np.ndarray,
    r_min: int,
    r_max: int,
    smooth_window: int,
) -> int:
    """Detect dominant FEP ring peak radius within a bounded search window."""
    if radial_profile.size == 0:
        return 255
    rp = _smooth_1d(radial_profile, window=smooth_window)
    lo = max(0, int(r_min))
    hi = min(int(r_max), rp.size - 1)
    if hi <= lo:
        return int(np.argmax(rp))
    return int(np.argmax(rp[lo:hi + 1]) + lo)


def evaluate_frame_with_status(
    filepath: Union[str, Path],
    frame_index: int = 0,
    mask: Optional[np.ndarray] = None,
    detector_path: Optional[str] = None,
    config: Optional[FrameMetricConfig] = None,
) -> tuple[Optional[FrameMetrics], FrameEvalStatus]:
    """Robust frame evaluation with structured status/error output."""
    cfg = config or FrameMetricConfig()
    filepath = Path(filepath)

    try:
        frame = _safe_load_pixium_frame(
            filepath,
            frame_index=frame_index,
            detector_path=detector_path,
            retries=cfg.load_retries,
            sleep_s=cfg.load_retry_sleep_s,
        )
    except Exception as exc:              # noqa: BLE001
        return None, FrameEvalStatus(
            filepath=str(filepath),
            frame_index=frame_index,
            status="error",
            error=f"load_failed: {exc}",
        )

    if mask is not None and mask.shape != frame.shape:
        return None, FrameEvalStatus(
            filepath=str(filepath),
            frame_index=frame_index,
            status="error",
            error=f"mask_shape_mismatch: mask={mask.shape}, frame={frame.shape}",
        )

    arr = frame.copy().astype(np.float64)
    if mask is not None:
        arr[mask.astype(bool)] = 0.0

    rows, cols = arr.shape
    cy, cx = rows / 2.0, cols / 2.0
    y_idx, x_idx = np.ogrid[:rows, :cols]
    r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(np.int32)
    r_max = int(r_map.max())

    radial_sum = np.bincount(r_map.ravel(), weights=arr.ravel(), minlength=r_max + 1)
    radial_cnt = np.bincount(r_map.ravel(), minlength=r_max + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial_mean = np.where(radial_cnt > 0, radial_sum / radial_cnt, 0.0)

    fep_ring_r = int(cfg.fixed_fep_ring_r) if cfg.fixed_fep_ring_r is not None else 255
    if cfg.detect_fep_ring or cfg.fixed_fep_ring_r is None:
        fep_ring_r = _auto_detect_fep_ring_r(
            radial_mean,
            r_min=cfg.fep_search_min_px,
            r_max=cfg.fep_search_max_px,
            smooth_window=cfg.smooth_window,
        )

    radial_background = radial_mean[r_map]
    residual = arr - radial_background

    outer = cfg.outer_excl_px if cfg.outer_excl_px is not None else r_max
    detection_mask = (r_map >= int(cfg.beam_excl_px)) & (r_map <= int(outer))
    if mask is not None:
        detection_mask &= ~mask.astype(bool)

    residual_valid = residual[detection_mask]
    if residual_valid.size == 0:
        return None, FrameEvalStatus(
            filepath=str(filepath),
            frame_index=frame_index,
            status="error",
            error="no_valid_pixels_after_masking",
            fep_ring_r=fep_ring_r,
        )

    noise_sigma = _mad_sigma(residual_valid) if cfg.use_mad_noise else float(residual_valid.std() + 1e-12)
    spot_threshold = float(cfg.sigma_multiplier) * noise_sigma
    spot_density = float(np.mean(residual_valid > spot_threshold))

    if spot_density >= cfg.crystal_threshold:
        auto_class = "crystal"
    elif spot_density <= cfg.solution_threshold:
        total_counts = float(arr.sum())
        auto_class = "gas" if total_counts < _gas_count_threshold(arr) else "solution"
    else:
        auto_class = "uncertain"

    # Reuse the legacy metric bundle for CSV compatibility.
    metrics = compute_frame_metrics(arr, filepath, frame_index=frame_index, mask=None)
    metrics.auto_class = auto_class
    metrics.crystal_score = float(np.clip(spot_density / max(cfg.crystal_threshold, 1e-12), 0.0, 1.0))

    return metrics, FrameEvalStatus(
        filepath=str(filepath),
        frame_index=frame_index,
        status="ok",
        auto_class=auto_class,
        crystal_score=metrics.crystal_score,
        spot_density=spot_density,
        noise_sigma=noise_sigma,
        spot_threshold=spot_threshold,
        fep_ring_r=fep_ring_r,
    )

def compute_frame_metrics(
    frame: np.ndarray,
    filepath: Union[str, Path],
    frame_index: int = 0,
    mask: Optional[np.ndarray] = None,
) -> FrameMetrics:
    """
    Compute classification metrics for a single 2D detector frame.

    Geometry calibrated for Diamond I11 EH2 Pixium detector based on
    observed radial profiles: main FEP ring at r~255px, second FEP
    feature at r~550px, detector half-width ~1400px.

    The key insight from real data: crystal hit and non-crystal frames
    have nearly identical radial profiles (same FEP rings, same diffuse
    scatter). The ONLY reliable discriminating signal is the presence of
    localised Bragg spots ABOVE the smooth radial background. These are
    detected by subtracting the radial mean from each pixel and looking
    for significant positive residuals outside the direct beam region.

    Parameters
    ----------
    frame :
        2D pixel array shape (rows, cols).
    filepath :
        Source file path (for labelling only).
    frame_index :
        Frame index within the source file.
    mask :
        Optional boolean array, same shape as frame.
        True = masked pixel (beamstop, hot pixel) -> excluded from all
        metrics. Pass your pyFAI mask (.npy) here to ignore the beamstop
        arm and hot pixels. If None, all pixels are used.

    Returns
    -------
    FrameMetrics
        Computed metrics plus heuristic crystal_score and auto_class.
        The human_label field is blank - fill during manual review.
    """
    filepath = Path(filepath)
    arr = frame.astype(np.float64)
    rows, cols = arr.shape

    # --- Apply mask ----------------------------------------------------------
    # Zero out beamstop, hot pixels, and any other masked regions before
    # computing any metric. This prevents the beamstop arm (which changes
    # position between beamtimes) from affecting classification.
    if mask is not None:
        arr = arr.copy()
        arr[mask.astype(bool)] = 0.0

    # --- Radial geometry -----------------------------------------------------
    # Build r_map from the geometric frame centre.
    # FEP ring confirmed at r~255px on I11 EH2 Pixium from real data.
    cy, cx = rows / 2.0, cols / 2.0
    y_idx, x_idx = np.ogrid[:rows, :cols]
    r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(np.int32)
    r_max = int(r_map.max())

    # --- Radial mean profile -------------------------------------------------
    # The smooth radial profile captures FEP rings + diffuse scatter.
    # Subtracting it from each pixel leaves only localised departures
    # (Bragg spots above the background, or noise below it).
    radial_sum = np.bincount(r_map.ravel(), weights=arr.ravel(), minlength=r_max + 1)
    radial_cnt = np.bincount(r_map.ravel(),                      minlength=r_max + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial_mean = np.where(radial_cnt > 0, radial_sum / radial_cnt, 0.0)

    # Residual image: each pixel minus its radial mean
    # Positive residuals = pixels brighter than the local ring average
    # This suppresses the FEP ring signal and reveals Bragg spots
    radial_background = radial_mean[r_map]
    residual = arr - radial_background

    # --- Basic statistics (on masked array) ----------------------------------
    valid_pixels  = arr[arr > 0]
    total_counts  = float(arr.sum())
    mean_counts   = float(valid_pixels.mean()) if len(valid_pixels) > 0 else 0.0
    spatial_variance = float(arr.var())

    # --- Spot detection (primary discriminating metric) ----------------------
    # Count pixels with significant positive residual outside the direct beam.
    # Direct beam / beamstop region: r < 60px (confirmed from profiles).
    # We use a threshold of 5x the local radial RMS as "significant".
    #
    # Crystal frame: sparse bright spots well above local background -> high count
    # Non-crystal:   smooth rings subtract cleanly, residual is mostly noise -> low count
    #
    # Exclude:
    #   r < 60px   : direct beam / beamstop shadow
    #   r > 1200px : outer detector edge, falling signal, edge artefacts
    detection_mask = (r_map >= 60) & (r_map <= 1200)

    residual_valid = residual[detection_mask]
    radial_bg_valid = radial_background[detection_mask]

    # Adaptive threshold: a pixel is a "spot" if its residual exceeds
    # 5x the local radial mean intensity. This scales with beam brightness
    # so it works across different exposure conditions.
    # Use std-based threshold — calibrated against glycine ground truth.
    # Background-relative threshold (5x|bg|) fails because bg values are
    # large (~400 counts) making the bar too high for any pixel to pass.
    residual_std   = float(residual_valid.std()) if len(residual_valid) > 0 else 1.0
    spot_threshold = 5.0 * residual_std
    spot_pixels    = np.sum(residual_valid > spot_threshold)
    n_valid        = detection_mask.sum()
    spot_density   = float(spot_pixels / n_valid) if n_valid > 0 else 0.0

    # --- Spot brightness (secondary metric) ----------------------------------
    # When spots are present, how bright are they relative to background?
    # A strong crystal hit has a few very bright spots; noise has many faint ones.
    if spot_pixels > 0:
        spot_residuals  = residual_valid[residual_valid > spot_threshold]
        bg_at_spots     = radial_bg_valid[residual_valid > spot_threshold]
        with np.errstate(invalid="ignore", divide="ignore"):
            spot_snr = float(np.median(spot_residuals / np.clip(bg_at_spots, 1, None)))
    else:
        spot_snr = 0.0

    # --- FEP ring azimuthal uniformity (tertiary metric) ---------------------
    # After radial subtraction, the FEP ring region should be near-zero for
    # all frames (crystal and non-crystal alike). Any residual azimuthal
    # variance in this region indicates either crystal spots on the ring
    # or a poorly centred beam. Kept as a supplementary feature for ML.
    #
    # FEP ring annulus: r = 220-290px (±35px around confirmed peak at 255px)
    fep_annulus = (r_map >= 220) & (r_map <= 290) & detection_mask
    if fep_annulus.any():
        phi_map  = np.arctan2(y_idx - cy, x_idx - cx)
        phi_vals = phi_map[fep_annulus]
        res_vals = residual[fep_annulus]
        n_bins   = 180
        phi_bins = np.floor(
            (phi_vals + np.pi) / (2 * np.pi) * n_bins
        ).astype(int)
        phi_bins = np.clip(phi_bins, 0, n_bins - 1)
        sec_sum  = np.bincount(phi_bins, weights=res_vals, minlength=n_bins)
        sec_cnt  = np.bincount(phi_bins,                   minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            sec_mean = np.where(sec_cnt > 0, sec_sum / sec_cnt, np.nan)
        valid_sec  = sec_mean[~np.isnan(sec_mean)]
        ring_score = float(np.std(valid_sec) / (mean_counts + 1e-9)) \
                     if len(valid_sec) > 1 else 0.0
    else:
        ring_score = 0.0

    # --- Legacy metrics (kept for ML feature completeness) -------------------
    # hotspot_ratio and radial_contrast are retained in the CSV as ML features
    # even though they are not the primary discriminators for this detector.
    med = float(np.median(arr[arr > 0])) if (arr > 0).any() else 1.0
    hotspot_ratio = float(np.mean(arr[detection_mask] > 5.0 * med))

    valid_r = radial_mean[radial_mean > 0]
    if len(valid_r) > 1:
        p50 = float(np.percentile(valid_r, 50))
        p90 = float(np.percentile(valid_r, 90))
        radial_contrast = float(p90 / p50) if p50 > 0 else 1.0
    else:
        radial_contrast = 1.0

    # low_angle_power: sum in r=60-200px (inside FEP ring, outside direct beam)
    low_angle_mask  = (r_map >= 60) & (r_map <= 200)
    low_angle_power = float(arr[low_angle_mask].sum()) if low_angle_mask.any() else 0.0

    # --- Crystal score (0-1) -------------------------------------------------
    # Primary signal: spot_density (fraction of valid pixels with residual
    # above N x local std after radial background subtraction).
    # Secondary: spot_snr.
    #
    # CALIBRATED on Diamond I11 glycine data (Run7_GLY_0.5VF_X2):
    #   50 confirmed crystal hits, 80 sampled misses, 5std threshold.
    #
    #   Crystal hits: spot_density mean=0.2106%  range=0.1583-0.2505%
    #   Misses:       spot_density mean=0.1953%  range=0.0061-0.2116%
    #
    #   Key finding: misses split into two populations:
    #     - Clean background (gas/empty): spot_density < 0.05%  <- ideal background
    #     - Near-hits (partial bolus):    spot_density ~ 0.20%  <- ambiguous, flag uncertain
    #
    #   A single threshold cannot separate all crystal from all miss because
    #   near-hit frames are physically ambiguous. Instead we use three zones:
    #     spot_density > 0.22%  -> crystal    (above miss maximum)
    #     spot_density < 0.05%  -> background (clean, safe to use as background)
    #     0.05% - 0.22%         -> uncertain  (manual review needed)
    #
    #   The background pool only uses clean background frames (<0.05%).
    #   This is conservative but ensures Bragg signal never contaminates
    #   the background average.
    sd_norm  = min(spot_density / 0.0012, 1.0)   # saturates at 0.12% (calibrated)
    snr_norm = min(spot_snr / 10.0, 1.0)

    crystal_score = float(np.clip(
        0.70 * sd_norm + 0.30 * snr_norm,
        0.0, 1.0,
    ))

    # --- Auto classification -------------------------------------------------
    # Three-zone classification calibrated on glycine ground truth data.
    # These thresholds are expressed as spot_density fractions (not %).
    #
    # Calibrated thresholds from glycine ground truth (Run7_GLY_0.5VF_X2):
    #   296 total frames, 50 confirmed crystal hits, 246 misses.
    #
    #   Key finding: crystal_score clusters tightly at 0.663-0.711 for ALL
    #   frames including near-hit misses — the glycine run was so heavily
    #   crystallising that ~140 miss frames contain partial crystal/nucleating
    #   solution indistinguishable from confirmed hits by this metric alone.
    #
    #   Clean background frames (score < 0.65): only 6 frames, zero crystal
    #   frames lost. Small but unambiguously clean — safe for background pool.
    #
    #   Classification strategy:
    #     crystal_score < 0.65  -> solution/gas  (6 clean BG frames, 0 crystal lost)
    #     crystal_score >= 0.65 -> uncertain     (everything else — review CSV)
    #
    #   NOTE: For this dataset type, ML classification is the correct long-term
    #   solution. The heuristic metric cannot separate crystal from near-hit
    #   frames. The uncertain frames go to CSV for manual labelling.
    #   For subtraction, the 6 clean BG frames are sufficient.
    CLEAN_BG_THRESHOLD = 0.65   # below this -> definitely clean background

    if crystal_score < CLEAN_BG_THRESHOLD and total_counts > 0:
        # Low score AND has counts -> clean background (not gas)
        if total_counts < _gas_count_threshold(arr):
            auto_class = "gas"
        else:
            auto_class = "solution"
    elif crystal_score >= CLEAN_BG_THRESHOLD:
        # Everything above threshold is either crystal or near-hit
        # Flag all as uncertain — manual review via CSV
        # process_dataset() with process_uncertain=True will process all of these
        auto_class = "uncertain"
    else:
        auto_class = "uncertain"

    return FrameMetrics(
        filepath          = str(filepath),
        frame_index       = frame_index,
        collection_number = _read_collection_number(filepath),
        total_counts      = total_counts,
        mean_counts       = mean_counts,
        spatial_variance  = spatial_variance,
        hotspot_ratio     = hotspot_ratio,
        radial_contrast   = radial_contrast,
        ring_score        = ring_score,
        low_angle_power   = low_angle_power,
        crystal_score     = crystal_score,
        auto_class        = auto_class,
    )


def _gas_count_threshold(arr: np.ndarray) -> float:
    """
    Frame-adaptive threshold for gas (N2) classification.
    Gas frames have very low counts everywhere — even their bright pixels
    are dim. Threshold: total counts < 5% of what a uniformly-lit frame
    at the 80th percentile would produce.
    """
    p80 = float(np.percentile(arr[arr > 0], 80)) if (arr > 0).any() else 0.0
    return 0.05 * p80 * arr.size


# =============================================================================
# Dataset classification
# =============================================================================

def classify_dataset(
    filepaths: Sequence[Union[str, Path]],
    detector_path: Optional[str] = None,
    csv_output: Optional[Union[str, Path]] = None,
    mask: Optional[np.ndarray] = None,
    config: Optional[FrameMetricConfig] = None,
    return_status: bool = False,
) -> Union[list[FrameMetrics], dict[str, Any]]:
    """
    Classify all frames across a dataset and export an ML-ready CSV.

    Run this first on any new dataset. It scores every frame, prints a
    summary, and writes a CSV with a blank human_label column for you
    to fill in during visual review. That CSV becomes your ML training
    dataset incrementally as you label more runs.

    Parameters
    ----------
    filepaths :
        All .nxs files in the dataset (crystal hits and misses together).
    detector_path :
        Override HDF5 detector path. Auto-detected if None.
    csv_output :
        Path for the CSV output. Defaults to first file's directory.
    mask :
        Optional pyFAI mask array (True = masked pixel). Pass your .npy
        mask file here to exclude beamstop, hot pixels etc. from all
        classification metrics. Strongly recommended — the beamstop arm
        position changes between beamtimes and affects spot detection.

        Load with: mask = np.load("X2_mask.npy")

    Returns
    -------
    list[FrameMetrics]
        One entry per frame, sorted by crystal_score descending.

    Examples
    --------
    >>> import glob, numpy as np
    >>> files = sorted(glob.glob("RAW_2D/Run1_DLM/*.nxs"))
    >>> mask = np.load("X2_Calib/X2_mask.npy")
    >>> metrics = classify_dataset(files, mask=mask, csv_output="Run1_classification.csv")
    >>> crystal = [m for m in metrics if m.auto_class == "crystal"]
    """
    all_metrics: list[FrameMetrics] = []
    status_rows: list[FrameEvalStatus] = []
    n_files = len(filepaths)
    cfg = config or FrameMetricConfig()

    if mask is not None:
        logger.info("Mask provided: %d pixels masked (%.1f%% of detector)",
                    int(mask.sum()), 100.0 * mask.sum() / mask.size)

    for i, fp in enumerate(filepaths, 1):
        fp = Path(fp)
        logger.info("[%d/%d] Classifying: %s", i, n_files, fp.name)
        try:
            with h5py.File(fp, "r") as f:
                ds_path = _find_detector_path(f, fp.name, detector_path)
                shape = f[ds_path].shape
            n_frames = shape[0] if len(shape) == 3 else 1
            for j in range(n_frames):
                m, status = evaluate_frame_with_status(
                    fp,
                    frame_index=j,
                    mask=mask,
                    detector_path=detector_path,
                    config=cfg,
                )
                status_rows.append(status)
                if m is not None:
                    all_metrics.append(m)
        except Exception as exc:          # noqa: BLE001
            logger.error("  Failed: %s — %s", fp.name, exc)
            status_rows.append(
                FrameEvalStatus(
                    filepath=str(fp),
                    frame_index=0,
                    status="error",
                    error=f"file_failed: {exc}",
                )
            )

    all_metrics.sort(key=lambda m: m.crystal_score, reverse=True)

    # Summary table
    counts = {c: sum(1 for m in all_metrics if m.auto_class == c)
              for c in ("crystal", "solution", "gas", "uncertain")}
    total = len(all_metrics)
    print(f"\n{'─'*64}")
    print(f"  Dataset classification  ({total} frames from {n_files} files)")
    print(f"{'─'*64}")
    for cls, n in counts.items():
        bar = "█" * int(40 * n / max(total, 1))
        print(f"  {cls:<12} {n:>4} frames  {bar}")
    print(f"{'─'*64}")

    crystal = [m for m in all_metrics if m.auto_class == "crystal"]
    if crystal:
        print("\n  Top crystal candidates:")
        for m in crystal[:10]:
            print(f"    score={m.crystal_score:.3f}  {Path(m.filepath).name}"
                  f"  frame={m.frame_index}")
    else:
        print("\n  No frames auto-classified as crystal.")
        print("  Review 'uncertain' frames in the CSV.")
    print()

    # CSV export
    if csv_output is None and filepaths:
        csv_output = Path(filepaths[0]).parent / "frame_classification.csv"
    if csv_output and all_metrics:
        _write_metrics_csv(all_metrics, Path(csv_output))
        print(f"  ML training CSV: {csv_output}\n")

    if return_status:
        return {
            "metrics": all_metrics,
            "status": status_rows,
        }
    return all_metrics


def _write_metrics_csv(metrics: list[FrameMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(metrics[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))
    logger.info("Wrote %d rows to %s", len(metrics), path)


# =============================================================================
# Background builder
# =============================================================================

def build_background_from_dataset(
    all_metrics: list[FrameMetrics],
    preferred_class: str = "solution",
    fallback_class: str = "gas",
    min_frames: int = 3,
    max_frames: int = 50,
    detector_path: Optional[str] = None,
) -> tuple[np.ndarray, str, int]:
    """
    Build the best available averaged background from non-crystal frames.

    Preference order:
      1. solution frames  (best match to crystal-hit scattering environment)
      2. gas frames       (cleanest, but leaves solution scatter in corrected frame)
      3. all non-crystal  (last resort with warning)

    Parameters
    ----------
    all_metrics :
        Output from classify_dataset.
    preferred_class :
        Frame class to prefer ("solution" recommended).
    fallback_class :
        Class to use if preferred frames are insufficient.
    min_frames :
        Minimum frames for a reliable average (warning if below).
    max_frames :
        Cap on number of frames; uses most background-like (lowest crystal_score).
    detector_path :
        HDF5 detector path override.

    Returns
    -------
    background : np.ndarray, shape (rows, cols)
    background_type : str
    n_frames : int
    """
    def _candidates(cls: str) -> list[FrameMetrics]:
        c = [m for m in all_metrics if m.auto_class == cls]
        c.sort(key=lambda m: m.crystal_score)   # most background-like first
        return c[:max_frames]

    preferred = _candidates(preferred_class)
    fallback  = _candidates(fallback_class)

    if len(preferred) >= min_frames:
        chosen   = preferred
        bg_label = preferred_class
    elif len(fallback) >= min_frames:
        logger.warning(
            "Only %d '%s' frames available; falling back to '%s' (%d frames).",
            len(preferred), preferred_class, fallback_class, len(fallback),
        )
        chosen   = fallback
        bg_label = fallback_class
    else:
        non_crystal = [
            m for m in all_metrics
            if m.auto_class not in ("crystal", "uncertain")
        ]
        non_crystal.sort(key=lambda m: m.crystal_score)
        chosen   = non_crystal[:max_frames]
        bg_label = "mixed_non_crystal"
        if len(chosen) < min_frames:
            warnings.warn(
                f"Only {len(chosen)} background frames available. "
                "Consider collecting dedicated FEP blank frames at the start of each run.",
                UserWarning, stacklevel=2,
            )

    if not chosen:
        raise RuntimeError(
            "No background frames found. All frames may be crystal or uncertain. "
            "Lower classification thresholds or provide a dedicated blank file."
        )

    logger.info(
        "Background: %d '%s' frames (noise factor 1/sqrt(%d) = %.3f)",
        len(chosen), bg_label, len(chosen), 1.0 / np.sqrt(len(chosen)),
    )

    stack: list[np.ndarray] = []
    for m in chosen:
        try:
            frame = _load_pixium_frame(m.filepath, m.frame_index, detector_path)
            if frame.ndim == 3:
                frame = frame[0]
            stack.append(frame.astype(np.float64))
        except Exception as exc:          # noqa: BLE001
            logger.warning("Could not load %s frame %d: %s",
                           Path(m.filepath).name, m.frame_index, exc)

    if not stack:
        raise RuntimeError("Failed to load any background frames from disk.")

    return np.mean(stack, axis=0), bg_label, len(stack)


def build_fep_background_2d(
    fep_filepath: Union[str, Path],
    frame_indices: Optional[Union[int, Sequence[int]]] = None,
    detector_path: Optional[str] = None,
) -> np.ndarray:
    """
    Build a background from a dedicated FEP blank file.

    This is the preferred future workflow: collect a dedicated blank
    (same flow, no crystallising material) at the start of each run.

    Parameters
    ----------
    fep_filepath :
        Dedicated FEP blank .nxs file.
    frame_indices :
        None = average all frames (recommended).

    Returns
    -------
    np.ndarray shape (rows, cols)
    """
    frames = _load_pixium_frame(fep_filepath, frame_indices, detector_path)
    if frames.ndim == 3:
        n  = frames.shape[0]
        bg = frames.mean(axis=0)
        logger.info(
            "Dedicated FEP background: %d frames averaged (noise / sqrt(%d) = %.3f)",
            n, n, 1.0 / np.sqrt(n),
        )
    else:
        bg = frames
        logger.info("Dedicated FEP background: single frame %s", bg.shape)
    return bg


# =============================================================================
# Core subtractor
# =============================================================================

def subtract_fep_2d(
    data_filepath: Union[str, Path],
    background: np.ndarray,
    output_filepath: Union[str, Path],
    background_type: str = "dataset_derived",
    n_background_frames: int = 0,
    *,
    frame_index: int = 0,
    data_detector_path: Optional[str] = None,
    scale_factor: Optional[float] = None,
    fep_reference_filepath: Optional[Union[str, Path]] = None,
    clip_negative: bool = True,
    output_detector_path: str = "entry1/pixium_hdf/data",
) -> SubtractionResult:
    """
    Subtract a 2D background from a single crystal data frame and write
    a corrected NeXus file ready for pyFAI azimuthal integration.

    Pixel-wise operation::

        corrected[i,j] = data[i,j] - (scale_factor * background[i,j])

    Parameters
    ----------
    data_filepath :
        Crystal-hit .nxs file.
    background :
        2D background array from build_background_from_dataset or
        build_fep_background_2d.
    output_filepath :
        Path for corrected output file.
    background_type :
        Label for provenance attributes in the output file.
    n_background_frames :
        Number of frames averaged to make the background (provenance).
    frame_index :
        Frame index to process from data file (default 0).
    data_detector_path :
        HDF5 detector path override.
    scale_factor :
        Override auto-scaling. None = auto from NeXus count_time.
    fep_reference_filepath :
        Reference file for exposure-time scaling (when scale_factor is None).
    clip_negative :
        Clip negative pixels to zero (default True).
    output_detector_path :
        HDF5 path for corrected data in output; must match what pyFAI expects.

    Returns
    -------
    SubtractionResult
    """
    data_filepath   = Path(data_filepath)
    output_filepath = Path(output_filepath)

    # Load data frame
    raw = _load_pixium_frame(data_filepath, frame_index, data_detector_path)
    if raw.ndim == 3:
        raw = raw[0]

    # Scale factor
    if scale_factor is not None:
        sf = float(scale_factor)
        logger.info("Manual scale_factor = %.6g", sf)
    elif fep_reference_filepath is not None:
        t_data = _read_exposure_time(data_filepath)
        t_fep  = _read_exposure_time(fep_reference_filepath)
        if t_data and t_fep and t_fep > 0:
            sf = t_data / t_fep
            logger.info("Auto scale: %.4f / %.4f = %.6g", t_data, t_fep, sf)
        else:
            sf = 1.0
            logger.warning("Exposure times unavailable; scale_factor = 1.0")
    else:
        sf = 1.0
        logger.info("No reference for scaling; scale_factor = 1.0")

    # Shape check
    if raw.shape != background.shape:
        raise ValueError(
            f"Shape mismatch — data: {raw.shape}, background: {background.shape}. "
            "Files must be from the same detector."
        )

    # Subtract
    corrected = raw - sf * background
    n_neg = int(np.sum(corrected < 0))
    pct   = 100.0 * n_neg / corrected.size
    logger.info(
        "Subtraction: scale=%.6g  negative pixels pre-clip: %d (%.2f%%)",
        sf, n_neg, pct,
    )
    if pct > 10.0:
        logger.warning(
            "%.1f%% negative pixels after subtraction. "
            "scale_factor may be too high, or background frames contain crystal signal.",
            pct,
        )
    if clip_negative:
        corrected = np.clip(corrected, 0.0, None)

    # Frame metrics of the input (for result bundle)
    metrics = compute_frame_metrics(raw, data_filepath, frame_index)

    # Write output
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    _write_corrected_nexus(
        corrected=corrected,
        source_filepath=data_filepath,
        output_filepath=output_filepath,
        output_detector_path=output_detector_path,
        scale_factor=sf,
        background_type=background_type,
        n_background_frames=n_background_frames,
        clip_negative=clip_negative,
    )
    logger.info("Written: %s", output_filepath)

    return SubtractionResult(
        corrected_frame     = corrected,
        scale_factor        = sf,
        background_type     = background_type,
        n_background_frames = n_background_frames,
        data_filepath       = str(data_filepath),
        output_filepath     = str(output_filepath),
        frame_metrics       = metrics,
    )


# =============================================================================
# Full pipeline (top-level entry point for automation)
# =============================================================================

def process_dataset(
    filepaths: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    dedicated_fep_filepath: Optional[Union[str, Path]] = None,
    scale_factor: Optional[float] = None,
    output_suffix: str = "_fep2d_corrected",
    csv_output: Optional[Union[str, Path]] = None,
    clip_negative: bool = True,
    detector_path: Optional[str] = None,
    process_uncertain: bool = False,
    mask: Optional[np.ndarray] = None,
    config: Optional[FrameMetricConfig] = None,
) -> dict:
    """
    Full pipeline: classify -> build background -> subtract -> save.

    This is the function to hand to the software engineer for inline
    automation. It accepts a list of files (e.g. from a folder watcher),
    classifies them, builds the best available background from non-crystal
    frames within the same dataset, and outputs corrected NeXus files
    for all crystal-hit frames.

    When dedicated blank frames become available at the start of each run,
    pass their file via dedicated_fep_filepath to use as background instead.

    Parameters
    ----------
    filepaths :
        All .nxs files to process (crystal hits and misses together).
    output_dir :
        Output directory for corrected files and CSV.
    dedicated_fep_filepath :
        Optional dedicated FEP blank file. If provided, used as background
        instead of deriving one from the dataset.
        Recommended future workflow: collect a blank at the start of each run.
    scale_factor :
        Manual scale override. None = auto from exposure time metadata.
    output_suffix :
        Appended to each output filename stem.
        e.g. i11-1-123456.nxs -> i11-1-123456_fep2d_corrected.nxs
    csv_output :
        Override path for the ML training CSV.
    clip_negative :
        Clip negative pixels (default True).
    detector_path :
        HDF5 detector path override.
    process_uncertain :
        If True, also process uncertain frames (borderline classification).
        Default False: uncertain frames go to CSV for manual review only.

    Returns
    -------
    dict:
        "processed"       : list[Path]         corrected files written
        "skipped"         : list[str]           non-crystal filenames
        "uncertain"       : list[str]           uncertain filenames for review
        "metrics"         : list[FrameMetrics]  all frame metrics
        "background_type" : str
        "n_bg_frames"     : int
        "csv_path"        : Path

    Examples
    --------
    Current workflow (background derived from dataset):

    >>> import glob
    >>> results = process_dataset(
    ...     sorted(glob.glob("RAW_2D/Run1_DLM/*.nxs")),
    ...     output_dir="Corrected_2D/Run1_DLM/",
    ... )

    Future workflow (dedicated blank at start of run):

    >>> results = process_dataset(
    ...     sorted(glob.glob("RAW_2D/Run1_DLM/*.nxs")),
    ...     output_dir="Corrected_2D/Run1_DLM/",
    ...     dedicated_fep_filepath="RAW_2D/FEP_blank/i11-1-122815.nxs",
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_output) if csv_output else output_dir / "frame_classification.csv"

    # Step 1: classify (mask applied here)
    print("Step 1/3 — Classifying frames...")
    all_metrics = classify_dataset(
        filepaths,
        detector_path=detector_path,
        csv_output=csv_path,
        mask=mask,
        config=config,
    )

    # Step 2: background
    print("Step 2/3 — Building background...")
    if dedicated_fep_filepath is not None:
        background  = build_fep_background_2d(
            dedicated_fep_filepath, detector_path=detector_path
        )
        bg_type     = "dedicated_fep_blank"
        n_bg_frames = 1
        fep_ref     = Path(dedicated_fep_filepath)
    else:
        background, bg_type, n_bg_frames = build_background_from_dataset(
            all_metrics, detector_path=detector_path
        )
        fep_ref = None
    print(f"  Background: {bg_type} ({n_bg_frames} frames averaged)\n")

    # Step 3: subtract crystal frames
    print("Step 3/3 — Subtracting background...")
    processed:  list[Path] = []
    skipped:    list[str]  = []
    uncertain:  list[str]  = []

    for m in all_metrics:
        fp  = Path(m.filepath)
        out = output_dir / f"{fp.stem}{output_suffix}{fp.suffix}"

        if m.auto_class == "crystal" or (
            process_uncertain and m.auto_class == "uncertain"
        ):
            try:
                subtract_fep_2d(
                    data_filepath          = fp,
                    background             = background,
                    output_filepath        = out,
                    background_type        = bg_type,
                    n_background_frames    = n_bg_frames,
                    frame_index            = m.frame_index,
                    data_detector_path     = detector_path,
                    scale_factor           = scale_factor,
                    fep_reference_filepath = fep_ref,
                    clip_negative          = clip_negative,
                )
                processed.append(out)
                print(f"  OK  {fp.name}  score={m.crystal_score:.3f}")
            except Exception as exc:          # noqa: BLE001
                logger.error("  FAIL  %s — %s", fp.name, exc)
        elif m.auto_class == "uncertain":
            uncertain.append(fp.name)
        else:
            skipped.append(fp.name)

    print(f"\n{'─'*64}")
    print(f"  Complete")
    print(f"  Corrected files : {len(processed)}")
    print(f"  Skipped         : {len(skipped)}")
    print(f"  Uncertain (CSV) : {len(uncertain)}")
    print(f"  Output dir      : {output_dir}")
    print(f"  CSV             : {csv_path}")
    print(f"{'─'*64}\n")

    return {
        "processed":       processed,
        "skipped":         skipped,
        "uncertain":       uncertain,
        "metrics":         all_metrics,
        "background_type": bg_type,
        "n_bg_frames":     n_bg_frames,
        "csv_path":        csv_path,
    }


# =============================================================================
# NeXus output writer
# =============================================================================

def _write_corrected_nexus(
    corrected: np.ndarray,
    source_filepath: Path,
    output_filepath: Path,
    output_detector_path: str,
    scale_factor: float,
    background_type: str,
    n_background_frames: int,
    clip_negative: bool,
) -> None:
    target_name   = output_detector_path.lstrip("/").split("/")[-1]
    target_parent = "/".join(output_detector_path.lstrip("/").split("/")[:-1])

    # Read all metadata from source into memory FIRST, then close source
    # before opening output. This avoids Windows/external-drive HDF5 locking
    # issues when two h5py files are open simultaneously.
    metadata: dict = {}  # path -> (data_or_None, attrs, is_group)

    def _collect(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if name.lstrip("/") == output_detector_path.lstrip("/"):
            return
        if isinstance(obj, h5py.Group):
            metadata[name] = (None, dict(obj.attrs), True)
        else:
            try:
                metadata[name] = (obj[()], dict(obj.attrs), False)
            except Exception as exc:          # noqa: BLE001
                logger.debug("Could not read /%s: %s", name, exc)

    root_attrs: dict = {}
    with h5py.File(source_filepath, "r") as src:
        root_attrs = dict(src.attrs)
        src.visititems(_collect)

    # Now write output with source fully closed
    with h5py.File(output_filepath, "w") as dst:
        # Restore root attributes
        for k, v in root_attrs.items():
            try:
                dst.attrs[k] = v
            except Exception as exc:          # noqa: BLE001
                logger.debug("Could not set root attr %s: %s", k, exc)

        # Recreate groups and datasets from collected metadata
        for name, (data, attrs, is_group) in metadata.items():
            try:
                if is_group:
                    grp = dst.require_group(name)
                    for k, v in attrs.items():
                        try:
                            grp.attrs[k] = v
                        except Exception:     # noqa: BLE001
                            pass
                else:
                    # Create parent groups if needed
                    parent = "/".join(name.split("/")[:-1])
                    if parent:
                        dst.require_group(parent)
                    ds_name = name.split("/")[-1]
                    parent_grp = dst[parent] if parent else dst
                    d = parent_grp.create_dataset(ds_name, data=data)
                    for k, v in attrs.items():
                        try:
                            d.attrs[k] = v
                        except Exception:     # noqa: BLE001
                            pass
            except Exception as exc:          # noqa: BLE001
                logger.debug("Could not write /%s: %s", name, exc)

        # Write corrected detector data
        grp = dst.require_group(target_parent) if target_parent else dst
        ds  = grp.create_dataset(
            target_name,
            data=corrected.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["long_name"]            = "FEP-background-subtracted Pixium frame"
        ds.attrs["background_type"]      = background_type
        ds.attrs["n_background_frames"]  = n_background_frames
        ds.attrs["scale_factor_applied"] = float(scale_factor)
        ds.attrs["negative_clipped"]     = bool(clip_negative)
        ds.attrs["processing_timestamp"] = datetime.datetime.utcnow().isoformat()
        ds.attrs["processing_script"]    = "fep_subtraction_2d_i11.py"
        ds.attrs["processing_note"] = (
            "2D FEP subtraction in pixel space before azimuthal integration. "
            "Ready for pyFAI integrate1d with existing .poni calibration."
        )
        if "NX_class" not in dst.attrs:
            dst.attrs["NX_class"] = "NXroot"


# =============================================================================
# Diagnostics
# =============================================================================

def inspect_nexus(filepath: Union[str, Path]) -> None:
    """Print full HDF5 tree with shapes. Run on a new file to confirm detector paths."""
    filepath = Path(filepath)
    print(f"\n{'='*64}\n  {filepath.name}\n{'='*64}")
    with h5py.File(filepath, "r") as f:
        def _p(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            indent = "  " + "    " * name.count("/")
            if isinstance(obj, h5py.Group):
                nx = obj.attrs.get("NX_class", b"")
                nx = nx.decode() if isinstance(nx, bytes) else nx
                print(f"{indent}D {name}" + (f"  [{nx}]" if nx else ""))
            else:
                print(f"{indent}  {name}   shape={obj.shape}  dtype={obj.dtype}")
        f.visititems(_p)
    print()


def quick_plot_subtraction(
    data_filepath: Union[str, Path],
    background: np.ndarray,
    scale_factor: float = 1.0,
    frame_index: int = 0,
) -> None:
    """
    Three-panel diagnostic: raw data | background | corrected.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        raise ImportError("pip install matplotlib")

    data_filepath = Path(data_filepath)
    frame = _load_pixium_frame(data_filepath, frame_index)
    if frame.ndim == 3:
        frame = frame[0]
    corrected = np.clip(frame - scale_factor * background, 0.0, None)

    vmin = max(1.0, float(np.percentile(frame[frame > 0], 1)))
    vmax = float(np.percentile(frame, 99.9))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = [
        f"Crystal data\n{data_filepath.name}",
        f"Background ({background.shape})",
        f"Corrected (scale={scale_factor:.4g}) — ready for pyFAI",
    ]
    for ax, img, lbl in zip(axes, [frame, background, corrected], labels):
        im = ax.imshow(img, origin="lower", norm=norm, cmap="viridis", aspect="auto")
        ax.set_title(lbl, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("2D FEP Subtraction — Diagnostic", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    corr_raw = frame - scale_factor * background
    n_neg    = int(np.sum(corr_raw < 0))
    print(f"\nDiagnostic summary")
    print(f"  scale_factor         : {scale_factor:.6g}")
    print(f"  data mean / max      : {frame.mean():.1f} / {frame.max():.0f}")
    print(f"  background mean / max: {background.mean():.1f} / {background.max():.0f}")
    print(f"  corrected mean / max : {corrected.mean():.1f} / {corrected.max():.0f}")
    print(f"  negative pixels      : {n_neg} ({100*n_neg/frame.size:.2f}%)")
    if n_neg / frame.size > 0.10:
        print("  WARNING: >10% negative. "
              "Reduce scale_factor or check background frame selection.")


# =============================================================================
# Entry point / example
# =============================================================================

if __name__ == "__main__":
    import glob
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Edit paths before running
    RUN_DIR    = r"E:/I11BT_dec25_dlm_gly/Data_Processing/RAW_2D/Run1_DLM_0.079VF_X3/"
    OUTPUT_DIR = r"E:/I11BT_dec25_dlm_gly/Data_Processing/Corrected_2D/Run1_DLM/"
    FEP_BLANK  = None  # e.g. r"E:/.../FEP_blank/i11-1-122815.nxs"

    all_files = sorted(glob.glob(RUN_DIR + "i11-1-*.nxs"))
    if not all_files:
        print(f"No files found in {RUN_DIR}", file=sys.stderr)
        sys.exit(1)

    # Always inspect a file first on a new dataset
    inspect_nexus(all_files[0])

    # Run the full pipeline
    results = process_dataset(
        filepaths              = all_files,
        output_dir             = OUTPUT_DIR,
        dedicated_fep_filepath = FEP_BLANK,
        process_uncertain      = False,
    )

    print("Corrected files:")
    for p in results["processed"]:
        print(f"  {p}")
    print(f"\nReview uncertain frames in: {results['csv_path']}")
