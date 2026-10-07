"""
fep2d_inline_1d_preview.py  -  OPTIONAL fast 1D preview, one call per frame
==========================================================================
NOT the inline subtraction script (that is fep2d_inline.py, 2D frame in, 2D frame out). This is the
earlier design that goes straight to a baselined 1D pattern without writing a 2D frame. Kept in case
a live 1D preview is wanted later. It uses fit_stride from Settings.
For use during an experiment: load once, then call process() on each new frame.

Why it is fast
--------------
Integration is linear, so   integrate(frame - s*ref) = integrate(frame) - s*integrate(ref)
when the same mask is used. The reference is integrated ONCE when the processor is
built. Each frame then costs one integration plus one scale fit (about 3.4 s), instead of a
full 2D subtraction, a write and a second integration. fit_stride=1 (default) is exact;
fit_stride=2 takes about 0.8 s but moves the pattern by up to 0.2 counts next to the FEP line.
fep2d_pipeline.py and the tests prove this equals the offline path.

Typical use
-----------
    from fep2d_inline_1d_preview import InlineProcessor
    import fep2d_v3 as fm, pyFAI, numpy as np

    ref  = fm.load_reference("merged_reference.npz")
    ai   = pyFAI.load("X3_CBZcalib.poni")
    mask = np.load("X3_CBZcalib_mask.npy")
    proc = InlineProcessor(ref, ai, mask)          # settings default to the locked ones
    res  = proc.process_file("pixium_146414.hdf")[0]
    res.x, res.y_corr          # corrected, baselined pattern
    res.ok, res.flags          # did it pass the checks, and what was noticed

Folder watching and saving
--------------------------
    python fep2d_inline_1d_preview.py --ref merged_reference.npz --poni X3.poni --mask mask.npy \
        --watch E:/run/frames --out E:/out --run Run7

Every frame gets a result, never an exception: a bad frame comes back with ok=False and
a reason in flags, so one bad file cannot stop an experiment.
"""
from __future__ import annotations

import argparse
import collections
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import fep2d_pipeline as fp
import fep2d_v3 as fm
from fep2d_pipeline import InputError, Settings


@dataclass
class FrameResult:
    name: str
    ok: bool
    flags: list = field(default_factory=list)
    x: Optional[np.ndarray] = None
    y_raw: Optional[np.ndarray] = None       # integrated, no subtraction
    y_sub: Optional[np.ndarray] = None       # subtracted, gap filled, not baselined
    y_corr: Optional[np.ndarray] = None      # subtracted, gap filled, baselined
    scale: float = float("nan")
    fit_frac: float = float("nan")
    noise: float = float("nan")              # robust far-angle noise of y_sub
    seconds: float = 0.0

    def summary(self) -> dict:
        return dict(name=self.name, ok=self.ok, flags=";".join(self.flags),
                    scale=round(self.scale, 4), fit_frac=round(self.fit_frac, 3),
                    noise=round(self.noise, 4), seconds=round(self.seconds, 3))


class InlineProcessor:
    def __init__(self, ref: "fm.Reference", ai, det_mask=None,
                 settings: Optional[Settings] = None):
        self.s = settings or Settings()
        self.ref, self.ai = ref, ai
        self.shape = tuple(ref.mean.shape)
        self.warnings = fp.check_reference(ref)          # raises InputError if unusable
        if ai is None:
            raise InputError("the inline processor needs the calibration (.poni)")
        self.mask = fp.total_mask(det_mask, ai, self.shape, self.s.extra_mask)
        st = max(int(self.s.fit_stride), 1)
        self._sl = (slice(None, None, st), slice(None, None, st))
        self._V = ~self.mask[self._sl]
        if self._V.sum() < 1000:
            raise InputError("fewer than 1000 usable pixels after masking")
        self._G = ref.mean[self._sl][self._V].astype(np.float64)
        self.x, self.i_ref = fp.integrate(ref.mean, ai, self.mask, self.s.npt,
                                          self.s.rmin, self.s.rmax)
        self._ref_total = float(ref.mean[~self.mask].sum())
        self._scales = collections.deque(maxlen=200)
        self._noises = collections.deque(maxlen=200)

    # ------------------------------------------------------------------
    def process(self, frame, name: str = "frame") -> FrameResult:
        t0 = time.time()
        r = FrameResult(name=name, ok=False)
        try:
            frame = np.asarray(frame)
            if frame.shape != self.shape:
                raise InputError(f"frame is {frame.shape} but the reference is {self.shape}")
            frame = frame.astype(np.float32)
            if not np.isfinite(frame).all():
                r.flags.append("non_finite_pixels")
                raise InputError("frame contains NaN or infinite pixels")
            total = float(frame[~self.mask].sum())
            if total < self.s.min_total_frac * self._ref_total:
                r.flags.append("low_total")
                raise InputError("frame total is far below the reference (shutter closed or dark?)")
            f = frame[self._sl][self._V].astype(np.float64)
            scale, frac = fp.fit_scale_fast(f, self._G)
            r.scale, r.fit_frac = scale, frac
            x, y_raw = fp.integrate(frame, self.ai, self.mask, self.s.npt, self.s.rmin, self.s.rmax)
            y_sub = fp.fill_gaps(x, y_raw - scale * self.i_ref, self.s.extra_mask)
            y_corr, _ = fp.baseline_mor(x, y_sub, self.s.half_window)
            if self.s.zero_masked:
                y_corr = fp.zero_ranges(x, y_corr, self.s.extra_mask)
            r.x, r.y_raw, r.y_sub, r.y_corr = x, y_raw, y_sub, y_corr
            r.noise = fp.far_noise(x, y_sub)
            r.ok = True
            self._soft_flags(r)
        except InputError as e:
            r.flags.append(f"rejected: {e}")
        except Exception as e:                          # never stop an experiment
            r.flags.append(f"error: {type(e).__name__}: {e}")
        r.seconds = time.time() - t0
        return r

    def _soft_flags(self, r: FrameResult):
        s = self.s
        lo, hi = s.scale_bounds
        if not (lo <= r.scale <= hi):
            r.flags.append(f"scale_out_of_range ({r.scale:.2f})")
        if r.fit_frac < s.min_fit_frac:
            r.flags.append(f"fit_unstable (kept {r.fit_frac:.2f} of pixels)")
        if len(self._scales) >= s.min_history:
            med = float(np.median(self._scales))
            mad = 1.4826 * float(np.median(np.abs(np.array(self._scales) - med))) + 1e-9
            if abs(r.scale - med) > s.drift_sigma * mad:
                r.flags.append(f"scale_drift (median {med:.3f})")
        if len(self._noises) >= s.min_history and np.isfinite(r.noise):
            nmed = float(np.median(self._noises))
            if nmed > 0 and r.noise > s.noise_factor * nmed:
                r.flags.append(f"high_noise ({r.noise:.3f} vs median {nmed:.3f})")
        # the history only learns from frames that passed the hard checks
        self._scales.append(r.scale)
        if np.isfinite(r.noise):
            self._noises.append(r.noise)

    def process_file(self, path) -> list:
        """One FrameResult per frame in the file (handles 2D or 3D datasets)."""
        out = []
        p = Path(path)
        try:
            for _, i, fr in fm.iter_frames([p]):
                out.append(self.process(fr, p.stem + (f"_f{i:04d}" if i else "")))
        except Exception as e:
            out.append(FrameResult(name=p.stem, ok=False,
                                   flags=[f"unreadable: {type(e).__name__}: {e}"]))
        return out

    # ------------------------------------------------------------------
    def corrected_2d(self, frame):
        """Full 2D corrected frame (only needed if you want to save 2D frames, e.g. for the
        classifier). Same maths as the offline app, using the inline scale fit."""
        corr, info = fm.subtract_frame(np.asarray(frame, np.float32), self.ref, self.mask)
        return corr, info


# ----------------------------------------------------------------------------
# saving and watching
# ----------------------------------------------------------------------------
def save_result(res: FrameResult, out_dir, save_baselined=True, save_subtracted=True):
    out = Path(out_dir)
    if not res.ok:
        return
    if save_subtracted:
        (out / "02_integrated").mkdir(parents=True, exist_ok=True)
        fp.save_xy(out / "02_integrated" / f"{res.name}.xy", res.x, res.y_sub,
                   "2theta(deg) Intensity_2Dcorr")
    if save_baselined:
        (out / "03_baselined").mkdir(parents=True, exist_ok=True)
        fp.save_xy(out / "03_baselined" / f"{res.name}_baselined.xy", res.x, res.y_corr,
                   "2theta(deg) Intensity_baselined")


def append_log(res: FrameResult, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res.summary().keys()))
        if new:
            w.writeheader()
        w.writerow(res.summary())


def _readable(p) -> bool:
    """True once the file opens as HDF5 (a file still being written usually does not)."""
    try:
        import h5py
        with h5py.File(p, "r"):
            return True
    except Exception:
        return False


def watch_folder(folder, exts=(".hdf",), poll=1.0, settle=0.5, stop=lambda: False, give_up_s=60.0):
    """Yield each NEW frame file once. A file is passed on when its size has stopped changing AND it
    opens as HDF5. If it still cannot be opened after give_up_s seconds it is passed on anyway, so it
    is logged as a failed frame instead of being silently skipped.
    Default is .hdf only: each .nxs links to its .hdf, so watching both counts every frame twice."""
    seen, first_seen, folder = set(), {}, Path(folder)
    exts = tuple(e.lower() for e in exts)
    while not stop():
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in exts and p not in seen:
                first_seen.setdefault(p, time.time())
                s1 = p.stat().st_size
                time.sleep(settle)
                if p.stat().st_size != s1 or s1 == 0:
                    continue
                if _readable(p) or time.time() - first_seen[p] > give_up_s:
                    seen.add(p)
                    yield p
        time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inline 2D FEP subtraction (folder watcher)")
    ap.add_argument("--ref", required=True, help="merged reference .npz")
    ap.add_argument("--poni", required=True)
    ap.add_argument("--mask", default=None, help="detector mask .npy (True = bad pixel)")
    ap.add_argument("--watch", required=True, help="folder to watch for new frames")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--run", default="run")
    ap.add_argument("--extra-mask", default=fp.fmt_ranges(fp.DEFAULT_EXTRA_MASK))
    ap.add_argument("--ext", default=".hdf")
    ap.add_argument("--half-window", type=int, default=4)
    a = ap.parse_args(argv)
    import pyFAI
    s = Settings(extra_mask=fp.parse_ranges(a.extra_mask), half_window=a.half_window)
    proc = InlineProcessor(fm.load_reference(a.ref), pyFAI.load(a.poni),
                           np.load(a.mask) if a.mask else None, s)
    for w in proc.warnings:
        print("WARNING:", w)
    out = Path(a.out) / a.run
    fp.write_manifest(out / "inline_manifest.json", reference=a.ref, poni=a.poni, mask=a.mask,
                      settings=s.as_dict(), reference_warnings=proc.warnings)
    print(f"watching {a.watch}  ->  {out}   (Ctrl+C to stop)")
    try:
        for p in watch_folder(a.watch, (a.ext,)):
            for res in proc.process_file(p):
                save_result(res, out)
                append_log(res, out / "inline_log.csv")
                print(res.name, "OK" if res.ok else "FAILED", f"{res.seconds:.2f}s", *res.flags)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
