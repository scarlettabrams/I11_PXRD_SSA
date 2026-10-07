"""
fep2d_pipeline.py  -  shared, headless building blocks for the 2D FEP pipeline
===============================================================================
Used by BOTH pipelines so they cannot drift apart:

    fep2d_app_v4.py   offline Streamlit app (calibration, reference, subtract,
                      integrate + baseline, pattern stacker)
    fep2d_inline.py   inline processor (one call per frame, quality flags,
                      folder watcher)

The 2D maths (reference building, per-frame scale fit, subtraction) lives in
fep2d_v3.py and is not changed. This file adds what comes after subtraction
and the settings that both pipelines share.

Settings locked on 07/10/26 (see the master notes, "Locked defaults"):
    extra 2-theta mask        5.30-6.10 deg  (removes the 5.40 spike and the
                              6.10 leftover; the CBZ 6.10 peak is lost)
    scale fit                 whole detector, no offset, robust, Bragg spots rejected
    integration               2000 points, 1-30 deg, no solid-angle correction
    gap across the mask       straight line between the two edge values
    baseline                  MOR (pybaselines), half_window 4, per frame
    merge                     plain mean of the baselined frames

Requires: numpy, h5py, pyFAI, pybaselines (and fep2d_v3.py beside this file).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import fep2d_v3 as fm

PIPELINE_VERSION = "fep2d_pipeline v4.0 (V2 maths via fep2d_v3 unchanged)"

DEFAULT_EXTRA_MASK = [(5.30, 6.10)]


# ----------------------------------------------------------------------------
# settings shared by both pipelines
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    extra_mask: list = field(default_factory=lambda: [tuple(r) for r in DEFAULT_EXTRA_MASK])
    npt: int = 2000
    rmin: float = 1.0
    rmax: float = 30.0
    half_window: int = 4
    zero_masked: bool = True            # set the masked window to 0 after baselining (see zero_ranges)
    # inline only: pixels used for the scale fit = every `fit_stride`-th pixel in both
    # directions. 1 = exact, same as the offline path (3.4 s/frame on the test machine);
    # 2 = a quarter of the pixels (0.8 s/frame, corrected pattern within 0.004 counts of exact);
    # 3 = a ninth (0.4 s/frame, within 0.02 counts). Measured on AP39 R007 frames.
    fit_stride: int = 2
    # inline quality checks
    scale_bounds: tuple = (0.3, 3.0)
    min_total_frac: float = 0.30        # frame total below this x reference total = dark / shutter
    min_fit_frac: float = 0.50          # fraction of pixels kept by the robust fit
    drift_sigma: float = 5.0            # scale this many robust sigma from the running median
    noise_factor: float = 2.0           # far-angle noise this many times the running median
    min_history: int = 10               # frames needed before drift / noise flags can fire

    def as_dict(self) -> dict:
        d = asdict(self)
        d["extra_mask"] = [list(r) for r in self.extra_mask]
        d["scale_bounds"] = list(self.scale_bounds)
        return d


class InputError(ValueError):
    """The frame, reference or geometry cannot be used. The message says why."""


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def parse_ranges(text: str):
    """'5.30-6.10, 8-8.5' -> [(5.3, 6.1), (8.0, 8.5)]. Raises ValueError."""
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"\s*([0-9.]+)\s*-\s*([0-9.]+)\s*", part)
        if not m:
            raise ValueError(f"range '{part}' not understood, write it like 5.30-6.10")
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi <= lo:
            raise ValueError(f"range '{part}' must have high > low")
        out.append((lo, hi))
    return out


def fmt_ranges(ranges) -> str:
    return ", ".join(f"{a:.2f}-{b:.2f}" for a, b in ranges)


def total_mask(det_mask, ai, shape, ranges) -> np.ndarray:
    """Detector mask OR the extra 2-theta annulus ranges. True = bad pixel."""
    total = np.zeros(shape, bool) if det_mask is None else np.asarray(det_mask).astype(bool).copy()
    if total.shape != tuple(shape):
        raise InputError(f"detector mask is {total.shape} but the frame is {tuple(shape)}")
    if ranges:
        if ai is None:
            raise InputError("an extra 2-theta mask needs the calibration (.poni)")
        total |= fm.tth_annulus_mask(ai, shape, ranges)
    return total


def integrate(frame, ai, mask, npt=2000, rmin=1.0, rmax=30.0):
    """Azimuthal integration, raw summed counts: no solid-angle correction, no error model."""
    import pyFAI
    import pyFAI.units
    res = ai.integrate1d(np.asarray(frame, np.float32), int(npt), unit=pyFAI.units.TTH_DEG,
                         radial_range=[rmin, rmax], mask=mask, correctSolidAngle=False,
                         error_model=None)
    return np.asarray(res.radial, float), np.asarray(res.intensity, float)


def fill_gaps(x, y, ranges):
    """Straight-line fill across each masked 2-theta range (same logic as the 1D app)."""
    y = np.asarray(y, float)
    for lo, hi in (ranges or []):
        y = fm.interpolate_gap(x, y, lo, hi)
    return y


def zero_ranges(x, y, ranges):
    """Set the masked 2-theta window(s) to exactly 0 (call AFTER baselining). The straight-line bridge
    is only there so the baseline has something to follow; any small bump left inside the window
    (about 0.4 counts seen at 6.05-6.10 deg on AP39 frames) is baseline edge noise, not data.
    The unbaselined pattern keeps the straight line."""
    y = np.asarray(y, float).copy()
    for lo, hi in (ranges or []):
        y[(np.asarray(x) >= lo) & (np.asarray(x) <= hi)] = 0.0
    return y


def baseline_mor(x, y, half_window=4):
    """MOR baseline (pybaselines). Returns (corrected, baseline)."""
    from pybaselines import Baseline
    base = Baseline(x_data=x).mor(np.asarray(y, float), half_window=int(half_window))[0]
    return np.asarray(y, float) - base, base


def far_noise(x, y, lo=22.5, hi=30.0) -> float:
    """Robust noise (MAD) of a pattern in a far-angle region with no strong peaks."""
    m = (x >= lo) & (x <= hi)
    if m.sum() < 5:
        return float("nan")
    v = np.asarray(y)[m]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def merge_patterns(xs: Sequence, ys: Sequence):
    """Mean and standard deviation of several patterns on the first pattern's grid."""
    if not len(ys):
        raise ValueError("nothing to merge")
    grid = np.asarray(xs[0], float)
    stack = np.array([np.interp(grid, np.asarray(x, float), np.asarray(y, float))
                      for x, y in zip(xs, ys)])
    return grid, stack.mean(axis=0), stack.std(axis=0)


def save_xy(path, x, y, header="2theta(deg) Intensity"):
    np.savetxt(path, np.column_stack([x, y]), header=header, comments="", fmt="%.6f")


def load_xy(path):
    d = np.loadtxt(path, skiprows=1, comments="#")
    if d.ndim < 2 or d.shape[1] < 2:
        raise ValueError(f"{path} is not a two-column .xy file")
    return d[:, 0], d[:, 1]


def fit_scale_fast(f, g, n_iter=6, clip=3.0):
    """Same robust scale fit as fep2d_v3.fit_scale (no offset), on already-selected pixels.
    f = frame pixels, g = reference pixels (float64, same order). Returns (scale, fraction kept)."""
    use = np.ones(f.size, bool)
    s = 1.0
    for _ in range(n_iter):
        s = float((f[use] * g[use]).sum() / ((g[use] ** 2).sum() + 1e-12))
        res = f - s * g
        sig = 1.4826 * np.median(np.abs(res[use] - np.median(res[use]))) + 1e-12
        use = (res < clip * sig) & (res > -clip * 2 * sig)
    return float(s), float(use.mean())


def check_reference(ref: "fm.Reference", min_frames=10, warn_frames=46) -> list:
    """Checks on a reference. Raises InputError for a reference that cannot be used;
    returns a list of warning strings for one that can but is weak."""
    warns = []
    if ref is None or ref.mean is None:
        raise InputError("no reference loaded")
    if not np.isfinite(ref.mean).all():
        raise InputError("the reference contains NaN or infinite pixels")
    if float(ref.mean.mean()) <= 0:
        raise InputError("the reference has a zero or negative mean level (empty or dark frames?)")
    srcs = [f"{p}::{i}" for p, i in ref.sources]
    if srcs and len(set(srcs)) != len(srcs):
        warns.append(f"the reference lists {len(srcs) - len(set(srcs))} duplicated frame(s)")
    names = [Path(p).stem for p, _ in ref.sources]
    nums = [re.search(r"(\d+)$", n).group(1) for n in names if re.search(r"(\d+)$", n)]
    if nums and len(set(nums)) != len(nums):
        warns.append(f"{len(nums) - len(set(nums))} exposure number(s) appear more than once "
                     "(an .nxs and its .hdf counted twice?). The true frame count is lower.")
    n_true = len(set(nums)) if nums else ref.n_used
    if n_true < min_frames:
        warns.append(f"only {n_true} distinct reference frames; fewer than {min_frames} cannot "
                     "find even the strong peaks reliably")
    elif n_true < warn_frames:
        warns.append(f"{n_true} reference frames: enough for strong peaks, but about "
                     f"{warn_frames} or more are needed for the weaker ones")
    return warns


def write_manifest(path, **items):
    items.setdefault("pipeline", PIPELINE_VERSION)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(items, fh, indent=2, default=str)
