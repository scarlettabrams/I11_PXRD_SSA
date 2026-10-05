"""
fep2d_v2.py
=============================================================
2D (pre-integration) background subtraction for I11 Pixium flow PXRD.
Rebuilt from scratch around the AP39 1D findings.

Design principles (from AP39)
-----------------------------
1. The reference is a MERGED frame: many background-only frames averaged
   to maximise SNR (AP39: 93 to 112 frames per reference; 5 frames was
   visibly too few).
2. Frame selection is NOT done here. Visual triage in the Streamlit app
   already labels diffraction / background / reference. This module only
   screens out physically bad frames (shutter closed, detector out of
   position) before they enter a reference.
3. Subtraction, not division (AP39: higher relative peak intensity, same
   peak positions).
4. The scale factor is fitted PER FRAME against the reference, because
   intensity drifts over a run. Bragg spots are rejected from the fit so
   they do not bias the scale.
5. Negative pixels are kept by default (clipping biases low-signal
   regions upward). pyFAI integrates negatives fine.

Typical use
-----------
    ref = build_reference(ref_paths, mask=mask, label="70:30 EtOH:H2O")
    save_reference(ref, "ref_70_30.npz")

    table = process_run(sample_paths, ref, "Corrected_2D/", mask=mask,
                        ai=ai, xy_dir="Integrated_2Dcorr/")

Requires: numpy, h5py. Optional: pyFAI (only for xy output).
"""
from __future__ import annotations

import csv
import datetime
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np

PathLike = Union[str, Path]

DETECTOR_PATHS = (
    "entry1/pixium_hdf/data",
    "entry/data/data",
    "entry1/data/data",
    "entry1/Pixium10:detector/data",
    "entry/Pixium10:detector/data",
    "entry1/detector/data",
    "entry/detector/data",
)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def _detector_path(f: h5py.File) -> str:
    for p in DETECTOR_PATHS:
        if p in f:
            return p
    found = []
    f.visititems(lambda n, o: found.append(n)
                 if isinstance(o, h5py.Dataset) and o.ndim >= 2 else None)
    if not found:
        raise RuntimeError("No 2D/3D dataset found")
    return max(found, key=lambda n: int(np.prod(f[n].shape[-2:])))


def iter_frames(paths: Sequence[PathLike]):
    """Yield (path, frame_index, 2D float32 array); handles 2D or 3D datasets."""
    for p in paths:
        with h5py.File(p, "r") as f:
            ds = f[_detector_path(f)]
            if ds.ndim == 2:
                yield str(p), 0, ds[()].astype(np.float32)
            else:
                for i in range(ds.shape[0]):
                    yield str(p), i, ds[i].astype(np.float32)


def _valid(mask: Optional[np.ndarray], shape) -> np.ndarray:
    """Boolean array, True = usable pixel. pyFAI convention: mask True = bad."""
    if mask is None:
        return np.ones(shape, bool)
    if mask.shape != shape:
        raise ValueError(f"mask {mask.shape} vs frame {shape}")
    return ~mask.astype(bool)


def _thumb(a: np.ndarray, b: int = 16) -> np.ndarray:
    """Block-mean downsample for cheap frame comparison."""
    r, c = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:r, :c].reshape(r // b, b, c // b, b).mean(axis=(1, 3))


# ----------------------------------------------------------------------------
# Reference building
# ----------------------------------------------------------------------------
@dataclass
class Reference:
    mean: np.ndarray                 # merged background, float32
    var_of_mean: np.ndarray          # per-pixel variance of the merged mean
    label: str
    n_used: int
    n_rejected: int
    sources: list = field(default_factory=list)
    rejected: list = field(default_factory=list)   # (file, idx, reason)


def screen_frames(paths, mask=None, min_total_frac=0.3, z_total=5.0, min_corr=0.97):
    """
    Pass 1: cheap screen. Returns (keep_list, reject_list).
    Reject if total counts are far below the median (shutter closed), are a
    robust outlier, or the thumbnail correlates poorly with the median
    thumbnail (detector out of position / different scene).
    """
    recs = []
    for p, i, fr in iter_frames(paths):
        v = _valid(mask, fr.shape)
        recs.append((p, i, float(fr[v].sum()), _thumb(np.where(v, fr, 0.0))))
    if not recs:
        raise RuntimeError("No frames found")

    totals = np.array([r[2] for r in recs])
    med = np.median(totals)
    mad = 1.4826 * np.median(np.abs(totals - med)) + 1e-12
    med_thumb = np.median(np.stack([r[3] for r in recs]), axis=0).ravel()
    med_thumb = (med_thumb - med_thumb.mean()) / (med_thumb.std() + 1e-12)

    keep, rej = [], []
    for (p, i, tot, th) in recs:
        t = th.ravel()
        corr = float(np.mean(((t - t.mean()) / (t.std() + 1e-12)) * med_thumb))
        if tot < min_total_frac * med:
            rej.append((p, i, "low_total (shutter closed?)"))
        elif abs(tot - med) > z_total * mad:
            rej.append((p, i, "total_outlier"))
        elif corr < min_corr:
            rej.append((p, i, f"scene_mismatch corr={corr:.3f}"))
        else:
            keep.append((p, i))
    return keep, rej


def build_reference(paths: Sequence[PathLike], mask=None, label="reference",
                    screen=True, **screen_kw) -> Reference:
    """Streaming mean + variance (Welford) so 100 frames never sit in RAM."""
    if screen:
        keep, rej = screen_frames(paths, mask, **screen_kw)
    else:
        keep, rej = [(p, i) for p, i, _ in iter_frames(paths)], []
    keep_set = set(keep)
    if not keep_set:
        raise RuntimeError("All frames rejected by screening")

    n, mean, m2 = 0, None, None
    for p, i, fr in iter_frames(paths):
        if (p, i) not in keep_set:
            continue
        fr = fr.astype(np.float64)
        if mean is None:
            mean, m2 = np.zeros_like(fr), np.zeros_like(fr)
        n += 1
        d = fr - mean
        mean += d / n
        m2 += d * (fr - mean)
    var = (m2 / max(n - 1, 1)) / n          # variance of the mean
    return Reference(mean.astype(np.float32), var.astype(np.float32), label,
                     n, len(rej), sorted(keep_set), rej)


def save_reference(ref: Reference, path: PathLike) -> None:
    np.savez_compressed(path, mean=ref.mean, var_of_mean=ref.var_of_mean,
                        label=ref.label, n_used=ref.n_used,
                        n_rejected=ref.n_rejected,
                        sources=np.array([f"{p}::{i}" for p, i in ref.sources]))


def load_reference(path: PathLike) -> Reference:
    z = np.load(path, allow_pickle=False)
    srcs = [tuple(s.rsplit("::", 1)) for s in z["sources"]]
    return Reference(z["mean"], z["var_of_mean"], str(z["label"]),
                     int(z["n_used"]), int(z["n_rejected"]),
                     [(a, int(b)) for a, b in srcs])


# ----------------------------------------------------------------------------
# Scale + subtraction
# ----------------------------------------------------------------------------
def fit_scale(frame, ref_mean, mask=None, n_iter=6, clip=3.0,
              r_range: Optional[tuple] = None, fit_offset=False,
              centre: Optional[tuple] = None):
    """
    Robust least-squares scale s (and optional offset) so frame ~ s*ref + c.
    Iteratively rejects pixels with large POSITIVE residuals (Bragg spots)
    and extreme negative ones, so crystal signal does not inflate s.
    r_range=(rmin, rmax) in pixels restricts the fit to an annulus
    (e.g. an FEP-dominated region); None = all valid pixels.
    centre=(row, col) is the beam centre in pixels (get it from the .poni
    via beam_centre_px(ai)); defaults to the frame centre.
    """
    v = _valid(mask, frame.shape)
    if r_range is not None:
        cy, cx = centre if centre is not None else \
            (frame.shape[0] / 2.0, frame.shape[1] / 2.0)
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        r = np.hypot(x - cx, y - cy)
        v &= (r >= r_range[0]) & (r <= r_range[1])
    f = frame[v].astype(np.float64)
    g = ref_mean[v].astype(np.float64)
    use = np.ones(f.size, bool)
    s, c = 1.0, 0.0
    for _ in range(n_iter):
        if fit_offset:
            A = np.column_stack([g[use], np.ones(use.sum())])
            (s, c), *_ = np.linalg.lstsq(A, f[use], rcond=None)
        else:
            s = float((f[use] * g[use]).sum() / ((g[use] ** 2).sum() + 1e-12))
        res = f - (s * g + c)
        sig = 1.4826 * np.median(np.abs(res[use] - np.median(res[use]))) + 1e-12
        use = (res < clip * sig) & (res > -clip * 2 * sig)
    return float(s), float(c), float(use.mean())


def beam_centre_px(ai) -> tuple:
    """(row, col) beam centre in pixels from a pyFAI AzimuthalIntegrator."""
    f2d = ai.getFit2D()
    return float(f2d["centerY"]), float(f2d["centerX"])


def subtract_frame(frame, ref: Reference, mask=None, scale: Optional[float] = None,
                   clip_negative=False, **fit_kw):
    s, c, frac_used = (scale, 0.0, 1.0) if scale is not None else \
        fit_scale(frame, ref.mean, mask, **fit_kw)
    out = frame - (s * ref.mean + c)
    if mask is not None:
        out = np.where(mask.astype(bool), 0.0, out)
    v = _valid(mask, frame.shape)
    neg_frac = float((out[v] < 0).mean())
    if clip_negative:
        out = np.clip(out, 0.0, None)
    info = dict(scale=s, offset=c, fit_pixel_frac=frac_used,
                neg_frac=neg_frac, corrected_sum=float(out[v].sum()))
    return out.astype(np.float32), info


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_corrected_nxs(src: PathLike, dst: PathLike, corrected, frame_index,
                        ref: Reference, info: dict) -> None:
    """Copy the source file, swap in the corrected 2D frame (single frame)."""
    shutil.copy2(src, dst)
    with h5py.File(dst, "r+") as f:
        p = _detector_path(f)
        attrs = dict(f[p].attrs)
        del f[p]
        ds = f.create_dataset(p, data=corrected[None, ...] if corrected.ndim == 2
                              else corrected, compression="gzip", compression_opts=4)
        for k, v in attrs.items():
            ds.attrs[k] = v
        ds.attrs.update(
            processing="fep2d_v2 pixel-wise reference subtraction",
            reference_label=ref.label, reference_n_frames=ref.n_used,
            scale_factor=info["scale"], offset=info["offset"],
            source_frame_index=frame_index,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )


def integrate_to_xy(corrected, ai, mask, out_path, npt=1000, rmin=1.0, rmax=30.0):
    """Same settings as the Streamlit app (no solid-angle correction)."""
    import pyFAI
    res = ai.integrate1d(corrected.astype(np.float32), npt,
                         unit=pyFAI.units.TTH_DEG, radial_range=[rmin, rmax],
                         mask=mask, correctSolidAngle=False, error_model=None)
    np.savetxt(out_path, np.column_stack([res.radial, res.intensity]),
               header="2theta(deg) Intensity_2Dcorr", comments="", fmt="%.6f")


def process_run(sample_paths: Sequence[PathLike], ref: Reference,
                out_dir: PathLike, mask=None, ai=None,
                xy_dir: Optional[PathLike] = None, write_nxs=True,
                npt=1000, rmin=1.0, rmax=30.0, **kw) -> list:
    """
    Subtract the reference from every frame in sample_paths (pass the
    triage 'diffraction' files). Writes a per-frame diagnostics CSV so
    scale-factor drift across the run can be plotted directly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if xy_dir:
        Path(xy_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for order, (p, i, fr) in enumerate(iter_frames(sample_paths)):
        corr, info = subtract_frame(fr, ref, mask, **kw)
        stem = Path(p).stem
        if write_nxs:
            write_corrected_nxs(p, out_dir / f"{stem}_2Dcorr.nxs", corr, i, ref, info)
        if xy_dir and ai is not None:
            integrate_to_xy(corr, ai, mask, Path(xy_dir) / f"{stem}_2Dcorr.xy",
                            npt, rmin, rmax)
        rows.append(dict(order=order, file=Path(p).name, frame=i, **info))
    if rows:
        with open(out_dir / "subtraction_diagnostics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows
