"""
inline_worker.py  -  background worker for the INLINE 2D PXRD pipeline
======================================================================
Run it in its own terminal (or start it from the app's sidebar). It is independent of Streamlit, so
clicking around in the app never slows it down and an app restart never stops it.

    python inline_worker.py --watch E:/AP40/subtracted --out E:/AP40/2D_output/Run7 ^
        --poni E:/calib/calib.poni --mask E:/calib/mask.npy ^
        --ml-dir E:/2D_sub_appV4_trial/ML_version1 --pipeline-dir C:/.../FEP_Subtraction

What it does, every --poll seconds:
  1. looks in the watch folder for new corrected frames, e.g.  I11-1-<collection number>_subtracted.npy  or
     i11-1-<number>.nxs / .hdf  (the suffix is optional; .npy, .hdf and .nxs are all read)
  2. only touches a file once it is complete (a .npy header says how big it must be; an HDF5 file must open and
     show its dataset; both must have stopped being modified), so a half-written frame is never read
  3. classifies it with the frozen classifier (Version 1.2), applies the per-frame guard
  4. integrates it to 1D with the mask and saves the .xy pattern
  5. appends one row to <out>/01b_sorting/ml_scores.csv
  6. adds it to a running merge if it is a hit (provisional; the final merge uses your sorting decisions)
  7. writes <out>/worker_heartbeat.json so the app can show that it is alive

Restart safe: on start it reads ml_scores.csv and skips every frame already done.
Stop it with Ctrl+C, or by creating a file called STOP in the run folder.

Files written under <out>/ (same layout as the offline app):
    01b_sorting/ml_scores.csv        one row per frame (this worker is the only writer)
    02_integrated/<frame>.xy         1D pattern, masked window bridged, not baselined
    04_merged/live_hits_sum.npz      running sum of hit frames (sum, count, files); only with --live-2d-sum
    worker_heartbeat.json, worker_manifest.json, worker.log
"""
import os
import sys


def _early_threads():
    """Cap the maths threads BEFORE numpy / xgboost load, so the app stays responsive on a laptop."""
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(v, sys.argv[i + 1])


_early_threads()

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="ignoring size because footprint is set")
import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

WORKER_VERSION = "inline_worker v3"
CSV_COLS = ["file", "collection", "score", "ml_label", "lower", "upper", "model", "version",
            "error", "saved", "guard", "sec"]
STOP = {"flag": False}


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(logfile, msg):
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    try:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# is the file finished being written?  and reading a frame
# ---------------------------------------------------------------------------
H5_PATHS = ("entry1/pixium_hdf/data", "entry/data/data", "entry1/data/data", "entry1/Pixium10:detector/data",
            "entry1/detector/data", "entry/detector/data")


def _h5_dataset_path(f):
    for p in H5_PATHS:
        if p in f:
            return p
    found = []
    f.visititems(lambda n, o: found.append(n) if hasattr(o, "ndim") and o.ndim >= 2 else None)
    if not found:
        raise RuntimeError("No 2D/3D dataset found")
    return max(found, key=lambda n: int(np.prod(f[n].shape[-2:])))


def read_frame_file(path):
    """One 2D float32 frame from .npy, or from .hdf / .h5 / .nxs (first frame of a 3D dataset)."""
    path = str(path)
    if path.lower().endswith(".npy"):
        a = np.load(path, allow_pickle=False)
    else:
        import h5py
        with h5py.File(path, "r") as f:
            a = f[_h5_dataset_path(f)][()]
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def npy_expected_size(path):
    """Bytes the file must have, from its own .npy header. Raises if the header is not complete yet."""
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        if ver == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif ver == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"unsupported .npy version {ver}")
        header_end = f.tell()
    if dtype.hasobject:
        raise ValueError("object array")
    return header_end + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def is_ready(path, settle_s):
    try:
        st = os.stat(path)
        if st.st_size == 0 or time.time() - st.st_mtime < settle_s:
            return False
        if str(path).lower().endswith(".npy"):
            return st.st_size == npy_expected_size(path)
        import h5py
        with h5py.File(path, "r") as f:                       # a half-written HDF5 file usually does not open
            return f[_h5_dataset_path(f)].shape is not None
    except Exception:
        return False


def diag_for(diag_csv, name, tries=4, wait=0.4):
    """scale / fit_pixel_frac / neg_frac the subtraction script logged for this frame (subtraction_diagnostics.csv in the
    watched folder, appended just after the frame appears). None if there is no such file or no row, in which case the
    guard still runs its frame-sanity and hot-pixel checks but not the run-relative ones."""
    stem = os.path.splitext(name)[0]
    for _ in range(tries):
        try:
            if os.path.isfile(diag_csv):
                import pandas as pd
                d = pd.read_csv(diag_csv)
                for r in d.itertuples():
                    s_ = os.path.splitext(str(r.file))[0] + ("_f%04d" % int(r.frame) if int(r.frame) else "")
                    if s_ == stem:
                        return dict(scale=float(r.scale), fit_pixel_frac=float(r.fit_pixel_frac), neg_frac=float(r.neg_frac))
            else:
                return None
        except Exception:
            pass
        time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# running merge of hit frames
# ---------------------------------------------------------------------------
class LiveMerge:
    """Running per-pixel sum of hit frames. Masked pixels are 0 in the frames and the mask is the same
    for every frame, so mean = sum / count and the masked pixels simply stay 0."""

    def __init__(self, path: Path):
        self.path, self.sum, self.files, self.dirty, self.last_save = path, None, [], False, time.time()
        if path.is_file():
            try:
                z = np.load(path, allow_pickle=False)
                self.sum, self.files = z["sum"], [str(s) for s in z["files"]]
            except Exception:
                self.sum, self.files = None, []

    def add(self, name, frame):
        if name in self.files:
            return
        if self.sum is None:
            self.sum = np.zeros(frame.shape, np.float64)
        self.sum += frame
        self.files.append(name)
        self.dirty = True

    def save(self, force=False, every_s=30.0):
        if self.sum is None or not self.dirty:
            return
        if not force and time.time() - self.last_save < every_s:
            return
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, sum=self.sum, count=np.int64(len(self.files)), files=np.array(self.files))
        os.replace(tmp, self.path)
        self.dirty, self.last_save = False, time.time()


# ---------------------------------------------------------------------------
def read_done(csv_path: Path):
    if not csv_path.is_file():
        return set()
    try:
        import pandas as pd
        return set(pd.read_csv(csv_path)["file"].astype(str))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", required=True, help="folder the subtracted .npy frames appear in")
    ap.add_argument("--out", required=True, help="run folder: <output folder>/<run name>")
    ap.add_argument("--poni", required=True)
    ap.add_argument("--mask", required=True, help="detector mask .npy (True = bad pixel)")
    ap.add_argument("--ml-dir", required=True, help="folder holding classify_v1.py, frame_guard.py, run_start_basic.py")
    ap.add_argument("--pipeline-dir", default=str(Path(__file__).resolve().parent),
                    help="folder holding fep2d_pipeline.py (default: next to this script)")
    ap.add_argument("--prefix", default="I11-1-", help="start of the file name (not case sensitive); empty = anything before the number")
    ap.add_argument("--suffix", default="_subtracted", help="optional end of the file name, before the extension")
    ap.add_argument("--ext", default="npy,hdf,nxs", help="file types to read, in order of preference if one frame exists in several")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between folder scans")
    ap.add_argument("--settle", type=float, default=1.0, help="a file must be untouched this long before it is read")
    ap.add_argument("--lower", type=float, default=None, help="override the lower cut-off (default: the model's config.json)")
    ap.add_argument("--upper", type=float, default=None, help="override the upper cut-off")
    ap.add_argument("--extra-mask", default="5.30-6.10", help='extra 2-theta mask, e.g. "5.30-6.10"; "" for none')
    ap.add_argument("--npt", type=int, default=2000)
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--rmax", type=float, default=30.0)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="(kept for old commands; the running 2D sum is now off unless --live-2d-sum is given)")
    ap.add_argument("--live-2d-sum", action="store_true", help="also keep a running 2D sum of hit frames in 04_merged/live_hits_sum.npz (about 66 MB, off by default)")
    ap.add_argument("--merge-save-s", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=2, help="max maths threads (keeps the laptop responsive)")
    ap.add_argument("--max-attempts", type=int, default=3)
    a = ap.parse_args()

    out = Path(a.out)
    d_sort, d_int, d_mrg = out / "01b_sorting", out / "02_integrated", out / "04_merged"
    for d in (d_sort, d_int, d_mrg):
        d.mkdir(parents=True, exist_ok=True)
    logfile = out / "worker.log"
    stop_file, hb_file, csv_path = out / "STOP", out / "worker_heartbeat.json", d_sort / "ml_scores.csv"
    if stop_file.exists():
        stop_file.unlink()
    watch = Path(a.watch)
    if not watch.is_dir():
        sys.exit(f"Watch folder not found: {watch}")
    if watch.resolve() == out.resolve():
        sys.exit("The watch folder and the run folder are the same. Use different folders.")

    for p in (a.pipeline_dir, a.ml_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fep2d_pipeline as fp
    from classify_v1 import ClassifierV1
    import pyFAI

    ai = pyFAI.load(a.poni)
    det = np.load(a.mask).astype(bool)
    ranges = fp.parse_ranges(a.extra_mask) if a.extra_mask.strip() else []
    clf = ClassifierV1(folder=a.ml_dir)
    lo = float(clf.lower if a.lower is None else a.lower)
    hi = float(clf.upper if a.upper is None else a.upper)
    if lo >= hi:
        sys.exit(f"Lower cut-off {lo} must be below upper cut-off {hi}.")
    model_name = getattr(clf, "model_name", "xgb")
    model_ver = getattr(clf, "rec", {}).get("version", "v1") if isinstance(getattr(clf, "rec", {}), dict) else "v1"

    rs = fg = guard = None
    if not a.no_guard:
        try:
            import run_start_basic as rs
            import frame_guard as fg
            guard = fg.RunGuard()
        except Exception as e:
            log(logfile, f"Guard could not be loaded ({e}); running without it.")

    atomic_write_text(out / "worker_manifest.json", json.dumps(dict(
        worker=WORKER_VERSION, started=now_iso(), watch=str(watch), out=str(out), poni=a.poni, mask=a.mask,
        ml_dir=a.ml_dir, model=model_name, model_version=model_ver, lower=lo, upper=hi,
        cutoffs_overridden=(a.lower is not None or a.upper is not None), extra_mask_2theta_deg=ranges,
        npt=a.npt, rmin=a.rmin, rmax=a.rmax, guard=guard is not None, prefix=a.prefix, suffix=a.suffix, extensions=a.ext,
        note="Frames are the subtraction script's corrected frames; masked pixels are 0 in them."), indent=2))

    exts = [e.strip().lower().lstrip(".") for e in a.ext.split(",") if e.strip()]
    pre = re.escape(a.prefix) if a.prefix else r".*?"
    suf = f"(?:{re.escape(a.suffix)})?" if a.suffix else ""
    pat = re.compile(rf"^{pre}(\d+){suf}\.({'|'.join(map(re.escape, exts))})$", re.IGNORECASE)
    diag_csv = str(watch / "subtraction_diagnostics.csv")
    done = read_done(csv_path)
    done_stems = {os.path.splitext(n)[0].lower() for n in done}
    merge = LiveMerge(d_mrg / "live_hits_sum.npz") if (a.live_2d_sum and not a.no_merge) else None  # LiveMerge(d_mrg / "live_hits_sum.npz")
    new_csv = not csv_path.is_file()
    attempts, recent, tmask_cache = {}, deque(maxlen=20), {}
    n_err, started = 0, now_iso()

    def handler(signum, frame):
        STOP["flag"] = True
    signal.signal(signal.SIGINT, handler)
    try:
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass

    def heartbeat(status, backlog, last=""):
        atomic_write_text(hb_file, json.dumps(dict(
            status=status, pid=os.getpid(), started=started, updated=now_iso(), updated_epoch=time.time(),
            processed_total=len(done), backlog=backlog, last_file=last, errors=n_err,
            sec_per_frame=round(float(np.mean(recent)), 3) if recent else None,
            hits_in_live_merge=len(merge.files) if merge else 0, lower=lo, upper=hi,
            model=model_name, model_version=model_ver, watch=str(watch), poll_s=a.poll)))

    log(logfile, f"Worker started. Watching {watch}. {len(done)} frames already done. cut-offs {lo}/{hi}, model {model_name} {model_ver}.")
    heartbeat("running", 0)

    while not STOP["flag"] and not stop_file.exists():
        cands = []
        for p in watch.iterdir():
            m = pat.match(p.name)
            if m and p.name not in done and os.path.splitext(p.name)[0].lower() not in done_stems:
                cands.append((int(m.group(1)), exts.index(m.group(2).lower()), p))
        cands.sort()
        cands = [(n, p) for n, _, p in cands]
        ready = [(n, p) for n, p in cands if is_ready(p, a.settle)]
        backlog = len(cands)
        if not ready:
            heartbeat("running", backlog)
            if merge:
                merge.save(every_s=a.merge_save_s)
            time.sleep(a.poll)
            continue

        num, p = ready[0]
        t0, err, gflag = time.time(), "", ""
        try:
            frame = read_frame_file(p)
            _, sc = clf.classify(frame)
            lab = "non_hit" if sc < lo else ("hit" if sc >= hi else "unsure")
            if guard is not None:
                try:
                    ok_g = rs.frame_sanity(frame)[0] == "OK"
                    fl = guard.check(frame, diag_for(diag_csv, p.name), sanity_ok=ok_g)
                    gflag = "+".join(fl)
                    if fl:
                        lab = "error" if "FAIL" in fl else fg.final_label(lab, fl)
                except Exception as e:
                    gflag = "guard_error"
            try:
                tm = tmask_cache.get(frame.shape)
                if tm is None:
                    tm = tmask_cache[frame.shape] = fp.total_mask(det, ai, frame.shape, ranges)
                x, y = fp.integrate(frame, ai, tm, a.npt, a.rmin, a.rmax)
                y = fp.fill_gaps(x, y, ranges)
                fp.save_xy(d_int / f"{p.stem}.xy", x, y, "2theta(deg) Intensity_2Dcorr")
            except Exception as e:
                err = f"integrate: {type(e).__name__}: {e}"[:150]
            if merge is not None and lab == "hit":
                merge.add(p.name, frame)
            row = dict(file=p.name, collection=num, score=round(float(sc), 5), ml_label=lab, lower=lo, upper=hi,
                       model=model_name, version=model_ver, error=err, saved=now_iso(), guard=gflag,
                       sec=round(time.time() - t0, 3))
        except Exception as e:
            attempts[p.name] = attempts.get(p.name, 0) + 1
            if attempts[p.name] < a.max_attempts:
                log(logfile, f"{p.name}: could not process ({type(e).__name__}: {e}); will retry "
                             f"({attempts[p.name]}/{a.max_attempts}).")
                time.sleep(a.poll)
                continue
            n_err += 1
            row = dict(file=p.name, collection=num, score="", ml_label="error", lower=lo, upper=hi,
                       model=model_name, version=model_ver, error=f"{type(e).__name__}: {e}"[:150],
                       saved=now_iso(), guard="", sec=round(time.time() - t0, 3))
            log(logfile, f"{p.name}: gave up after {a.max_attempts} attempts, logged as error.")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if new_csv:
                w.writeheader()
                new_csv = False
            w.writerow(row)
        done.add(p.name)
        done_stems.add(os.path.splitext(p.name)[0].lower())
        recent.append(row["sec"])
        log(logfile, f"{p.name}  score {row['score']}  {row['ml_label']}  {row['guard']}  {row['sec']} s  (backlog {backlog - 1})")
        if merge:
            merge.save(every_s=a.merge_save_s)
        heartbeat("running", backlog - 1, p.name)

    if merge:
        merge.save(force=True)
    heartbeat("stopped", 0)
    log(logfile, "Worker stopped.")


if __name__ == "__main__":
    main()
