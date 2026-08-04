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
import json
import numpy as np
from scipy import stats

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
class LearnedThresholds:
    """
    Thresholds learned from a dataset's own spot_density distribution via GMM.
    Produced by learn_thresholds() and passed into classify_dataset().
    """
    crystal_threshold:       float
    background_threshold:    float
    separation_score:        float
    bic_ratio:               float
    n_crystal_estimated:     int
    n_background_estimated:  int
    fit_converged:           bool
    fallback_used:           bool
    crystal_mean:            float
    crystal_std:             float
    background_mean:         float
    background_std:          float
    n_frames_fitted:         int
    dataset_label:           str = ""
    timestamp:               str = ""

    def to_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Thresholds saved: %s", path)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "LearnedThresholds":
        with open(path) as f:
            return cls(**json.load(f))


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
    """Runtime-tunable parameters for robust frame scoring.

    threshold_mode controls which thresholds are used:
      'auto'     - use LearnedThresholds passed into classify_dataset()
      'manual'   - use crystal_threshold / solution_threshold set here
      'fallback' - use V0 glycine-calibrated values (with warning)
    """
    beam_excl_px:        int            = 60
    outer_excl_px:       Optional[int]  = 1200
    sigma_multiplier:    float          = 5.0
    use_mad_noise:       bool           = True
    detect_fep_ring:     bool           = True
    fixed_fep_ring_r:    Optional[int]  = None
    fep_search_min_px:   int            = 80
    fep_search_max_px:   int            = 600
    smooth_window:       int            = 11
    crystal_threshold:   float          = 0.22 / 100.0
    solution_threshold:  float          = 0.05 / 100.0
    load_retries:        int            = 2
    load_retry_sleep_s:  float          = 0.2
    threshold_mode:      str            = "fallback"
    min_frames_for_gmm:  int            = 50


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
# V1 — Automatic threshold learning via Gaussian Mixture Model
# =============================================================================

_GMM_MIN_SEPARATION = 0.5
_GMM_MIN_BIC_RATIO  = 1.05


def _fit_gmm_1d(x: np.ndarray) -> dict:
    """Fit a two-component GMM to 1D data using EM. Returns component params + BICs."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    mu_1  = x.mean()
    sig_1 = x.std() + 1e-12
    ll_1  = float(np.sum(stats.norm.logpdf(x, mu_1, sig_1)))
    bic_1 = -2 * ll_1 + 1 * np.log(n)

    med  = float(np.median(x))
    lo   = x[x <= med]; hi = x[x > med]
    mu0  = float(lo.mean()) if len(lo) > 1 else float(x.min())
    mu1  = float(hi.mean()) if len(hi) > 1 else float(x.max())
    sig0 = float(lo.std()) + 1e-12
    sig1 = float(hi.std()) + 1e-12
    w0 = w1 = 0.5

    prev_ll = -np.inf
    converged = False
    for _ in range(300):
        p0 = w0 * stats.norm.pdf(x, mu0, sig0)
        p1 = w1 * stats.norm.pdf(x, mu1, sig1)
        denom = p0 + p1 + 1e-300
        r0 = p0 / denom; r1 = p1 / denom
        n0 = r0.sum() + 1e-9; n1 = r1.sum() + 1e-9
        mu0  = float((r0 * x).sum() / n0)
        mu1  = float((r1 * x).sum() / n1)
        sig0 = float(np.sqrt((r0 * (x - mu0)**2).sum() / n0)) + 1e-12
        sig1 = float(np.sqrt((r1 * (x - mu1)**2).sum() / n1)) + 1e-12
        w0   = float(n0 / (n0 + n1)); w1 = float(n1 / (n0 + n1))
        ll   = float(np.sum(np.log(p0 + p1 + 1e-300)))
        if abs(ll - prev_ll) < 1e-6:
            converged = True; break
        prev_ll = ll

    if mu0 > mu1:
        mu0, mu1   = mu1, mu0
        sig0, sig1 = sig1, sig0
        w0, w1     = w1, w0

    ll_2  = float(np.sum(np.log(
        w0 * stats.norm.pdf(x, mu0, sig0) +
        w1 * stats.norm.pdf(x, mu1, sig1) + 1e-300
    )))
    bic_2 = -2 * ll_2 + 4 * np.log(n)

    return dict(mu0=mu0, sig0=sig0, w0=w0,
                mu1=mu1, sig1=sig1, w1=w1,
                bic_1=bic_1, bic_2=bic_2, converged=converged)


def _find_valley(x_grid: np.ndarray, pdf0: np.ndarray, pdf1: np.ndarray) -> float:
    """Find the minimum between two GMM component PDFs."""
    combined = pdf0 + pdf1
    peak0 = x_grid[np.argmax(pdf0)]; peak1 = x_grid[np.argmax(pdf1)]
    lo, hi = min(peak0, peak1), max(peak0, peak1)
    mask   = (x_grid >= lo) & (x_grid <= hi)
    if mask.sum() < 3:
        return float((lo + hi) / 2.0)
    sub_x = x_grid[mask]
    return float(sub_x[np.argmin(combined[mask])])


def learn_thresholds(
    filepaths: Sequence[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    mask: Optional[np.ndarray] = None,
    detector_path: Optional[str] = None,
    config: Optional[FrameMetricConfig] = None,
    dataset_label: str = "",
    save_plot: bool = True,
    max_frames_to_score: int = 500,
) -> LearnedThresholds:
    """
    Learn classification thresholds from a dataset's own spot_density
    distribution using a two-component Gaussian mixture model (GMM).

    Call this ONCE per dataset before running classify_dataset().
    Saves thresholds.json and threshold_diagnostic.png to output_dir.

    Parameters
    ----------
    filepaths :
        All .nxs / .hdf files in the dataset.
    output_dir :
        Where to save thresholds.json and diagnostic plot.
        Defaults to the directory of the first file.
    mask :
        Optional pyFAI mask (True = masked pixel). Strongly recommended.
    detector_path :
        HDF5 detector path override. Auto-detected if None.
    config :
        FrameMetricConfig for scoring parameters.
    dataset_label :
        Human-readable label for provenance (e.g. run name).
    save_plot :
        Save threshold_diagnostic.png. Requires matplotlib.
    max_frames_to_score :
        Cap on frames used for GMM fitting (uniformly sampled).

    Returns
    -------
    LearnedThresholds
        Pass directly to classify_dataset(learned_thresholds=...).

    Examples
    --------
    >>> thresh = learn_thresholds(files, output_dir="Run3_CBZ/",
    ...                           mask=mask, dataset_label="Run3_CBZ")
    >>> metrics = classify_dataset(files, mask=mask,
    ...                            learned_thresholds=thresh)
    """
    cfg = config or FrameMetricConfig()
    filepaths = [Path(f) for f in filepaths]

    if output_dir is None:
        output_dir = filepaths[0].parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build flat list of (filepath, frame_index) across all files
    frame_refs: list[tuple[Path, int]] = []
    for fp in filepaths:
        try:
            with h5py.File(fp, "r") as f:
                ds_path = _find_detector_path(f, fp.name, detector_path)
                shape   = f[ds_path].shape
            n_frames = shape[0] if len(shape) == 3 else 1
            for j in range(n_frames):
                frame_refs.append((fp, j))
        except Exception as exc:   # noqa: BLE001
            logger.warning("Could not inspect %s: %s", fp.name, exc)

    total_frames = len(frame_refs)
    if total_frames == 0:
        raise RuntimeError("No frames found in provided filepaths.")

    if total_frames < cfg.min_frames_for_gmm:
        warnings.warn(
            f"Only {total_frames} frames available for GMM fitting "
            f"(recommended minimum: {cfg.min_frames_for_gmm}). "
            "Learned thresholds may be unreliable — inspect the diagnostic plot.",
            UserWarning, stacklevel=2,
        )

    if total_frames > max_frames_to_score:
        indices    = np.round(np.linspace(0, total_frames - 1,
                                          max_frames_to_score)).astype(int)
        frame_refs = [frame_refs[i] for i in indices]

    print(f"\nlearn_thresholds: scoring {len(frame_refs)} frames "
          f"(from {total_frames} total) ...")

    all_spot_densities: list[float] = []
    for i, (fp, fidx) in enumerate(frame_refs, 1):
        _, status = evaluate_frame_with_status(
            fp, frame_index=fidx, mask=mask,
            detector_path=detector_path, config=cfg,
        )
        if status.status == "ok":
            all_spot_densities.append(float(status.spot_density))
        if i % 50 == 0 or i == len(frame_refs):
            print(f"  Scored {i}/{len(frame_refs)} frames")

    if len(all_spot_densities) < 10:
        raise RuntimeError(
            f"Only {len(all_spot_densities)} frames scored. "
            "Check file paths and detector path."
        )

    x = np.array(all_spot_densities, dtype=np.float64)
    n_scored = len(x)
    print(f"\n  {n_scored} frames scored.  "
          f"mean={x.mean():.5f}  std={x.std():.5f}  "
          f"min={x.min():.5f}  max={x.max():.5f}")

    gmm       = _fit_gmm_1d(x)
    bic_ratio = gmm["bic_1"] / (gmm["bic_2"] + 1e-12)
    separation = (abs(gmm["mu1"] - gmm["mu0"])
                  / (gmm["sig0"] + gmm["sig1"] + 1e-12))

    print(f"  GMM: converged={gmm['converged']}  "
          f"BIC ratio={bic_ratio:.3f}  separation={separation:.3f}")
    print(f"  Background: mu={gmm['mu0']:.5f}  sig={gmm['sig0']:.5f}  "
          f"w={gmm['w0']:.2f}")
    print(f"  Crystal:    mu={gmm['mu1']:.5f}  sig={gmm['sig1']:.5f}  "
          f"w={gmm['w1']:.2f}")

    use_fallback = (
        not gmm["converged"]
        or bic_ratio < _GMM_MIN_BIC_RATIO
        or separation < _GMM_MIN_SEPARATION
    )

    if use_fallback:
        warnings.warn(
            f"GMM did not find clean separation "
            f"(separation={separation:.3f}, BIC ratio={bic_ratio:.3f}). "
            "Using V0 glycine-calibrated fallback thresholds.",
            UserWarning, stacklevel=2,
        )
        crystal_threshold    = cfg.crystal_threshold
        background_threshold = cfg.solution_threshold
    else:
        x_grid = np.linspace(x.min(), x.max(), 2000)
        pdf0   = gmm["w0"] * stats.norm.pdf(x_grid, gmm["mu0"], gmm["sig0"])
        pdf1   = gmm["w1"] * stats.norm.pdf(x_grid, gmm["mu1"], gmm["sig1"])
        crystal_threshold    = _find_valley(x_grid, pdf0, pdf1)
        background_threshold = max(0.0, gmm["mu0"] - 1.5 * gmm["sig0"])
        print(f"\n  Learned thresholds:")
        print(f"    crystal    > {crystal_threshold:.5f} ({crystal_threshold*100:.4f}%)")
        print(f"    background < {background_threshold:.5f} ({background_threshold*100:.4f}%)")

    thresholds = LearnedThresholds(
        crystal_threshold      = float(crystal_threshold),
        background_threshold   = float(background_threshold),
        separation_score       = float(separation),
        bic_ratio              = float(bic_ratio),
        n_crystal_estimated    = int(round(gmm["w1"] * n_scored)),
        n_background_estimated = int(round(gmm["w0"] * n_scored)),
        fit_converged          = bool(gmm["converged"]),
        fallback_used          = bool(use_fallback),
        crystal_mean           = float(gmm["mu1"]),
        crystal_std            = float(gmm["sig1"]),
        background_mean        = float(gmm["mu0"]),
        background_std         = float(gmm["sig0"]),
        n_frames_fitted        = n_scored,
        dataset_label          = dataset_label,
        timestamp              = datetime.datetime.utcnow().isoformat(),
    )

    json_path = output_dir / "thresholds.json"
    thresholds.to_json(json_path)

    if save_plot:
        _plot_threshold_diagnostic(x, gmm, thresholds,
                                   output_dir / "threshold_diagnostic.png",
                                   dataset_label)

    print(f"\n  Saved: {json_path}")
    if save_plot:
        print(f"  Saved: {output_dir / 'threshold_diagnostic.png'}")
    print(f"  Est. crystal frames    : {thresholds.n_crystal_estimated}")
    print(f"  Est. background frames : {thresholds.n_background_estimated}")
    print(f"{'─'*60}\n")
    return thresholds


def _plot_threshold_diagnostic(
    spot_densities: np.ndarray,
    gmm: dict,
    thresholds: "LearnedThresholds",
    output_path: Path,
    dataset_label: str = "",
) -> None:
    """Save a diagnostic plot of the GMM fit and learned thresholds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — diagnostic plot skipped.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pct = spot_densities * 100.0
    ax.hist(pct, bins=60, density=True, alpha=0.4, color="steelblue",
            label=f"Observed ({len(pct)} frames)")

    x_grid = np.linspace(pct.min(), pct.max(), 1000)
    pdf0 = gmm["w0"] * stats.norm.pdf(x_grid, gmm["mu0"]*100, gmm["sig0"]*100)
    pdf1 = gmm["w1"] * stats.norm.pdf(x_grid, gmm["mu1"]*100, gmm["sig1"]*100)
    ax.plot(x_grid, pdf0, "--", color="coral",  lw=1.5,
            label=f"Background (mu={gmm['mu0']*100:.4f}%)")
    ax.plot(x_grid, pdf1, "--", color="teal",   lw=1.5,
            label=f"Crystal    (mu={gmm['mu1']*100:.4f}%)")
    ax.plot(x_grid, pdf0 + pdf1, "-", color="black", lw=2.0, label="GMM combined")

    ct = thresholds.crystal_threshold    * 100
    bt = thresholds.background_threshold * 100
    ax.axvline(ct, color="teal",  lw=2, ls=":", label=f"Crystal threshold {ct:.4f}%")
    ax.axvline(bt, color="coral", lw=2, ls=":", label=f"BG threshold {bt:.4f}%")
    ax.axvspan(pct.min(), bt, alpha=0.08, color="coral")
    ax.axvspan(ct, pct.max(),  alpha=0.08, color="teal")

    quality = "FALLBACK" if thresholds.fallback_used else "GOOD"
    color   = "red"      if thresholds.fallback_used else "green"
    ax.text(0.98, 0.97,
            f"Separation: {thresholds.separation_score:.2f}\n"
            f"BIC ratio:  {thresholds.bic_ratio:.2f}\nStatus: {quality}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.8))

    title = "Threshold diagnostic"
    if dataset_label:
        title += f" — {dataset_label}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Spot density (%)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Diagnostic plot saved: %s", output_path)


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
    learned_thresholds: Optional[LearnedThresholds] = None,
) -> Union[list[FrameMetrics], dict[str, Any]]:
    """
    Classify all frames across a dataset and export an ML-ready CSV.

    Parameters
    ----------
    filepaths :
        All .nxs / .hdf files in the dataset.
    detector_path :
        Override HDF5 detector path. Auto-detected if None.
    csv_output :
        Path for the CSV output. Defaults to first file's directory.
    mask :
        Optional pyFAI mask array (True = masked pixel).
    config :
        FrameMetricConfig. Threshold fields overridden if
        learned_thresholds is provided.
    learned_thresholds :
        Output of learn_thresholds(). If provided, overrides static
        threshold values. If None, uses V0 glycine-calibrated fallback.

    Returns
    -------
    list[FrameMetrics], sorted by crystal_score descending.
    """
    all_metrics: list[FrameMetrics] = []
    status_rows: list[FrameEvalStatus] = []
    n_files = len(filepaths)
    cfg = config or FrameMetricConfig()

    if learned_thresholds is not None:
        cfg.crystal_threshold  = learned_thresholds.crystal_threshold
        cfg.solution_threshold = learned_thresholds.background_threshold
        cfg.threshold_mode     = "auto"
        logger.info(
            "Using learned thresholds: crystal=%.5f  background=%.5f  "
            "(separation=%.2f  fallback=%s)",
            learned_thresholds.crystal_threshold,
            learned_thresholds.background_threshold,
            learned_thresholds.separation_score,
            learned_thresholds.fallback_used,
        )
    else:
        logger.warning(
            "No learned_thresholds provided — using V0 glycine-calibrated "
            "fallback (crystal=%.5f, background=%.5f). "
            "Run learn_thresholds() first for reliable cross-system classification.",
            cfg.crystal_threshold, cfg.solution_threshold,
        )

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
# Option 2 — Radial background synthesised from crystal frames
# =============================================================================

def build_radial_background_from_crystal_frames(
    all_metrics: list,
    beam_centre_rc: Optional[tuple] = None,
    mask: Optional[np.ndarray] = None,
    detector_path: Optional[str] = None,
    smooth_window: int = 21,
    max_frames: int = 50,
) -> tuple:
    """
    Build a 2D background image by synthesising the radial mean profile
    from classified crystal-hit frames.

    A crystal-hit frame is composed of two overlapping signals:
      (1) a smooth, rotationally symmetric FEP + solution background
      (2) sparse, localised Bragg spots

    Because Bragg spots cover only ~0.2% of pixels, they barely affect
    the azimuthal average at any given radius.  The radial mean profile
    is therefore almost entirely determined by the FEP background.

    Pooling the radial profiles from multiple crystal frames and
    reconstructing a 2D background image from the average profile
    gives a background that is:
      - derived from frames with the same material in the beam path
        as the crystal hit (same solution, same flow conditions)
      - lower noise than a single-frame estimate due to averaging
      - free of any requirement for dedicated blank frames

    Parameters
    ----------
    all_metrics :
        Output from classify_dataset(). Crystal frames are extracted
        automatically.
    beam_centre_rc :
        (row, col) beam centre in pixels.  If None, uses geometric
        frame centre (rows/2, cols/2).  Pass the true centre from
        your pyFAI .poni file for best results:
            import pyFAI
            ai = pyFAI.load("X2_calib.poni")
            cy, cx = ai.getFit2D()["centerY"], ai.getFit2D()["centerX"]
            beam_centre_rc = (cy, cx)
    mask :
        Optional pyFAI mask array (True = masked pixel).  Masked
        pixels are excluded from the radial average.
    detector_path :
        HDF5 detector path override.
    smooth_window :
        Moving-average window for smoothing the master radial profile
        before reconstruction.  Larger values give a smoother
        background but may blur real radial features.  Default 21.
    max_frames :
        Maximum number of crystal frames to use.  Frames are sorted
        by crystal_score descending (strongest hits first) and capped
        at this number.  Default 50.

    Returns
    -------
    background : np.ndarray, shape (rows, cols)
        Synthesised 2D background ready for subtract_fep_2d().
    background_type : str
        Label for provenance attributes.
    n_frames : int
        Number of crystal frames used.

    Notes
    -----
    The main limitation is that Bragg spots will slightly bias the
    radial mean upward at their positions.  For typical glycine data
    (spot density ~0.2%) this effect is small.  Option 3 (per-frame
    with spot masking) addresses this directly.
    """
    # Select crystal frames, strongest hits first, capped at max_frames
    crystal = [m for m in all_metrics if m.auto_class == "crystal"]
    # Also include uncertain frames that score highly if crystals are few
    if len(crystal) < 5:
        uncertain = [m for m in all_metrics if m.auto_class == "uncertain"]
        crystal = crystal + uncertain
        logger.warning(
            "Fewer than 5 crystal frames found; including uncertain frames "
            "(%d total) for radial background estimation.", len(crystal)
        )
    crystal.sort(key=lambda m: m.crystal_score, reverse=True)
    chosen = crystal[:max_frames]

    if not chosen:
        raise RuntimeError(
            "No crystal or uncertain frames available for radial background. "
            "Cannot synthesise background without at least one classified frame."
        )

    logger.info(
        "Radial background: using %d crystal frames (max_frames=%d)",
        len(chosen), max_frames,
    )

    # Load first frame to establish geometry
    first = _safe_load_pixium_frame(
        chosen[0].filepath,
        frame_index=chosen[0].frame_index,
        detector_path=detector_path,
    )
    rows, cols = first.shape

    # Beam centre
    if beam_centre_rc is not None:
        cy, cx = float(beam_centre_rc[0]), float(beam_centre_rc[1])
        logger.info("Using provided beam centre: row=%.2f  col=%.2f", cy, cx)
    else:
        cy, cx = rows / 2.0, cols / 2.0
        logger.warning(
            "No beam_centre_rc provided; using geometric centre (%.1f, %.1f). "
            "Pass the true centre from your .poni file for best results.", cy, cx
        )

    # Build integer radius map
    y_idx, x_idx = np.ogrid[:rows, :cols]
    r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(np.int32)
    r_max = int(r_map.max())

    # Accumulate weighted radial sums across all chosen frames
    radial_sum_total = np.zeros(r_max + 1, dtype=np.float64)
    radial_cnt_total = np.zeros(r_max + 1, dtype=np.float64)
    n_loaded = 0

    mask_flat = mask.astype(bool).ravel() if mask is not None else None

    for m in chosen:
        try:
            frame = _safe_load_pixium_frame(
                m.filepath,
                frame_index=m.frame_index,
                detector_path=detector_path,
            )
        except Exception as exc:   # noqa: BLE001
            logger.warning(
                "Could not load %s frame %d for radial BG: %s",
                Path(m.filepath).name, m.frame_index, exc
            )
            continue

        arr = frame.astype(np.float64)

        # Build per-frame valid pixel mask
        valid = np.ones(rows * cols, dtype=bool)
        if mask_flat is not None:
            valid &= ~mask_flat
        # Exclude zero pixels (dead pixels / outside detector)
        valid &= arr.ravel() > 0

        r_flat   = r_map.ravel()
        arr_flat = arr.ravel()

        radial_sum_total += np.bincount(
            r_flat[valid], weights=arr_flat[valid], minlength=r_max + 1
        )
        radial_cnt_total += np.bincount(
            r_flat[valid], minlength=r_max + 1
        )
        n_loaded += 1

    if n_loaded == 0:
        raise RuntimeError("Failed to load any crystal frames for radial background.")

    logger.info("Loaded %d / %d crystal frames for radial profile.", n_loaded, len(chosen))

    # Compute master radial mean profile
    with np.errstate(invalid="ignore", divide="ignore"):
        master_profile = np.where(
            radial_cnt_total > 0,
            radial_sum_total / radial_cnt_total,
            0.0,
        )

    # Fill any zero-count radii by linear interpolation
    zero_mask = radial_cnt_total == 0
    if zero_mask.any():
        r_axis = np.arange(r_max + 1)
        good   = ~zero_mask
        master_profile[zero_mask] = np.interp(
            r_axis[zero_mask], r_axis[good], master_profile[good]
        )

    # Smooth the profile to reduce per-frame statistical noise
    master_profile_smooth = _smooth_1d(master_profile, window=smooth_window)

    # Reconstruct 2D background: each pixel gets the profile value at its radius
    background = master_profile_smooth[r_map]

    logger.info(
        "Radial background synthesised: shape=%s  mean=%.1f  max=%.1f  "
        "n_frames=%d  smooth_window=%d",
        background.shape, background.mean(), background.max(),
        n_loaded, smooth_window,
    )

    return background, "radial_crystal_frames", n_loaded


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
    background_mode: str = "pool",
    beam_centre_rc: Optional[tuple] = None,
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
    if dedicated_fep_filepath is not None or background_mode == "dedicated":
        fep_path = dedicated_fep_filepath
        background  = build_fep_background_2d(fep_path, detector_path=detector_path)
        bg_type     = "dedicated_fep_blank"
        n_bg_frames = 1
        fep_ref     = Path(fep_path)
    elif background_mode == "radial_crystal":
        background, bg_type, n_bg_frames = build_radial_background_from_crystal_frames(
            all_metrics,
            beam_centre_rc = beam_centre_rc,
            mask           = mask,
            detector_path  = detector_path,
        )
        fep_ref = None
    else:
        # Default: "pool" — current behaviour
        background, bg_type, n_bg_frames = build_background_from_dataset(
            all_metrics, detector_path=detector_path
        )
        fep_ref = None
    print(f"  Background: {bg_type} ({n_bg_frames} frames used)\n")

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
