#!/usr/bin/env python
"""
Rebuild "inline processing" evidence from a SAVED run folder produced by
app_ssa_V3_beamline.py, for use in a beamtime proposal.

What it does
------------
1. Latency (acquisition -> integrated pattern), from file timestamps:
     - acquisition time  ~ mtime of  <run>/i11-1-<n>.nxs
       (the app copies with shutil.copy2, which preserves mtime, so this is
        the time the beamline finished writing the file)
     - integration time  ~ mtime of  <run>/<name>_01_integrated/i11-1-<n>_integrated.xy
     - latency           = integration time - acquisition time
   Also reports copy time (acq -> copied) when the OS exposes file creation
   time (Windows).
2. Waterfall of all integrated patterns (2theta vs frame/time).
3. Peak tracker: area of the most-changing reflection vs frame/time
   (or the reflection you specify with --peak).
4. Pipeline schematic with the measured numbers filled in.
5. Optional: per-frame integration compute time (--poni/--mask), which is
   the compute-limited figure, free of any human click time.

Usage
-----
    python beamtime_demo_figures.py "D:/my_user_dir/run_042"
    python beamtime_demo_figures.py RUN --peak 8.4 --peak-halfwidth 0.12
    python beamtime_demo_figures.py RUN --poni calib.poni --mask mask.npy

Outputs go to <RUN>/demo_figures/ (or --out).

IMPORTANT caveats (the script warns about these automatically where it can):
  * If the run folder was moved/zipped/copied after the beamtime by something
    that resets modification times, latency cannot be recovered.
  * If frames were re-integrated in bulk (Batch tab, or "reprocess everything"),
    the .xy mtimes reflect that later run, not the beamtime.
  * In the Inline tab integration is button-triggered, so latency includes
    the time taken to click. It is an end-to-end figure, not pure compute.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ── discovery ───────────────────────────────────────────────────────────────
def collect(run_folder: Path, integrated_dir: Path = None):
    nxs = {}
    for p in run_folder.glob("i11-1-*.nxs"):
        m = re.search(r"i11-1-(\d+)\.nxs$", p.name)
        if m:
            nxs[int(m.group(1))] = p

    if integrated_dir is None:
        candidates = [d for d in run_folder.glob("*_01_integrated") if d.is_dir()]
        if not candidates:
            sys.exit(f"No *_01_integrated folder found in {run_folder}. "
                     "Use --integrated-dir to point at it.")
        integrated_dir = max(candidates, key=lambda d: len(list(d.glob("*.xy"))))
        if len(candidates) > 1:
            print(f"[info] several *_01_integrated folders; using {integrated_dir.name}")

    xy = {}
    for p in integrated_dir.glob("*_integrated.xy"):
        m = re.search(r"i11-1-(\d+)_integrated\.xy$", p.name)
        if m:
            xy[int(m.group(1))] = p
    return nxs, xy, integrated_dir


def load_xy(path):
    a = np.loadtxt(path, skiprows=1)
    return a[:, 0], a[:, 1]


# ── latency ─────────────────────────────────────────────────────────────────
def build_table(nxs, xy):
    ids = sorted(set(nxs) & set(xy))
    rows = []
    for i in ids:
        acq = nxs[i].stat().st_mtime
        integ = xy[i].stat().st_mtime
        copied = float("nan")
        if sys.platform.startswith("win"):
            c = nxs[i].stat().st_ctime          # creation time on Windows = copy time
            if c >= acq - 1:
                copied = c
        rows.append(dict(collection=i, acq=acq, copied=copied, integrated=integ,
                         latency_s=integ - acq,
                         copy_latency_s=(copied - acq) if copied == copied else float("nan")))
    return rows


def sanity_checks(rows):
    warns = []
    if len(rows) < 3:
        warns.append("Fewer than 3 frames have both a .nxs and an integrated .xy.")
        return warns
    acq = np.array([r["acq"] for r in rows])
    integ = np.array([r["integrated"] for r in rows])
    lat = integ - acq
    if np.ptp(acq) < 2 and len(rows) > 3:
        warns.append("Acquisition timestamps are all within ~2 s of each other: modification "
                     "times were probably reset when the folder was moved. Latency is NOT recoverable.")
    if np.ptp(acq) > 60 and np.ptp(integ) < 0.1 * np.ptp(acq):
        warns.append("Integrated files were all written in a short burst compared with the "
                     "acquisition span: this looks like a bulk (Batch / reprocess) run, not inline. "
                     "Do not quote these as inline latencies.")
    if (lat < 0).mean() > 0.05:
        warns.append(f"{(lat < 0).mean():.0%} of latencies are negative: clocks or timestamps are inconsistent.")
    return warns


def summarise(rows):
    lat = np.array([r["latency_s"] for r in rows])
    lat = lat[lat >= 0]
    if lat.size == 0:
        return {}
    out = dict(n=lat.size, median=np.median(lat),
               q25=np.percentile(lat, 25), q75=np.percentile(lat, 75),
               p90=np.percentile(lat, 90), max=lat.max(),
               within_30s=(lat <= 30).mean(), within_60s=(lat <= 60).mean(),
               within_5min=(lat <= 300).mean())
    cl = np.array([r["copy_latency_s"] for r in rows])
    cl = cl[~np.isnan(cl) & (cl >= 0)]
    if cl.size:
        out["copy_median"] = np.median(cl)
    return out


def fmt_s(s):
    if s < 90:
        return f"{s:.0f} s"
    return f"{s/60:.1f} min"


# ── optional: pure compute time ─────────────────────────────────────────────
def time_integration(nxs, poni, mask_path, n=10, npt=1000, rmin=1.0, rmax=30.0, frame=None):
    """Mirror of the app's load_nxs_frame + integrate_frame, timed.

    If `frame` (a collection number) is given, that single frame is timed
    n times in a row instead of the first n distinct frames — useful for
    getting a steady-state number off one specific, known-good frame.
    """
    import h5py
    import pyFAI
    ai = pyFAI.load(poni)
    mask = np.load(mask_path) if mask_path else None
    if frame is not None:
        if frame not in nxs:
            sys.exit(f"--frame {frame} not found among .nxs files in the run folder "
                     f"(looked for i11-1-{frame}.nxs).")
        files = [nxs[frame]] * n
    else:
        files = [nxs[k] for k in sorted(nxs)][:n]
    times = []
    for f in files:
        t0 = time.perf_counter()
        with h5py.File(f, "r") as h:
            data = np.array(h["/entry1/pixium_hdf/data"][()][:])
        frame = data.reshape(data.shape[1:]).astype(np.float32)
        ai.integrate1d(frame, npt, unit=pyFAI.units.TTH_DEG,
                       radial_range=[rmin, rmax], mask=mask,
                       correctSolidAngle=False, error_model=None)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    # first call includes pyFAI's one-off engine set-up: report both
    return dict(first=times[0], steady_median=float(np.median(times[1:])) if len(times) > 1 else times[0],
                n=len(times))


# ── stack / peak tracker ────────────────────────────────────────────────────
def build_stack(xy, order):
    x0, y0 = load_xy(xy[order[0]])
    M = np.empty((len(order), x0.size))
    for k, i in enumerate(order):
        x, y = load_xy(xy[i])
        M[k] = y if (x.size == x0.size and np.allclose(x, x0)) else np.interp(x0, x, y)
    return x0, M


_trapz = getattr(np, "trapezoid", None) or np.trapz


def peak_area(x, M, centre, hw):
    sel = (x >= centre - hw) & (x <= centre + hw)
    xs = x[sel]
    areas = []
    for row in M:
        ys = row[sel]
        bg = np.linspace(ys[0], ys[-1], ys.size)      # linear background across window
        areas.append(_trapz(ys - bg, xs))
    return np.array(areas)


def auto_peak(x, M, hw):
    """Pick the reflection that changes most across the run (so static FEP
    tubing peaks are not chosen just because they are strong)."""
    sd = np.std(M, axis=0)
    k = np.argmax(sd)
    return float(x[k])


# ── plotting ────────────────────────────────────────────────────────────────
def draw_schematic(ax, stats, compute):
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    auto_c, man_c = "#d5ecd9", "#fbe3c4"
    boxes = [
        (2,  "Detector\n(Pixium, I11)", auto_c),
        (22, "Raw beamline dir\n(read-only)\n.nxs + .hdf", auto_c),
        (42, "User run folder\nauto-copy\n(8 s poll)", auto_c),
        (62, "pyFAI integration\n1D .xy per frame", man_c),
        (82, "Live plot / triage\n(Streamlit app)", man_c),
    ]
    w, h, y = 16, 12, 12
    for x, label, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=col, ec="0.3", lw=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.5)
    for x0, x1 in [(18, 22), (38, 42), (58, 62), (78, 82)]:
        ax.add_patch(FancyArrowPatch((x0 + 0.6, y + h / 2), (x1 - 0.6, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=10, color="0.3"))

    # measured labels underneath
    if "copy_median" in stats:
        ax.text(40, 8.0, f"acq → copied\nmedian {fmt_s(stats['copy_median'])}",
                ha="center", va="top", fontsize=7, color="#1a5c2e")
    if compute:
        ax.text(70, 8.0, f"compute ≈ {compute['steady_median']:.2f} s/frame\n(steady state)",
                ha="center", va="top", fontsize=7, color="#7a4a00")
    if stats:
        ax.annotate("", xy=(90, 5.2), xytext=(10, 5.2),
                    arrowprops=dict(arrowstyle="<->", color="0.2", lw=1))
        ax.text(50, 3.4,
                f"end-to-end (acquired → integrated): median {fmt_s(stats['median'])}, "
                f"IQR {fmt_s(stats['q25'])}–{fmt_s(stats['q75'])}, "
                f"n = {stats['n']} frames",
                ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.text(2, 28.5, "■ automated", color="#3f8f55", fontsize=7.5, va="center")
    ax.text(16, 28.5, "■ user-triggered (button)", color="#c98a2b", fontsize=7.5, va="center")


def make_figure(rows, stats, compute, x, M, order, peak, hw, out_png, highlight=None):
    acq = np.array([r["acq"] for r in rows])
    t_ok = np.ptp(acq) > 2
    t_min = (acq - acq.min()) / 60.0 if t_ok else np.arange(len(rows), dtype=float)
    t_label = "Time since first frame (min)" if t_ok else "Frame (collection order)"

    fig = plt.figure(figsize=(11, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.6, 1.0])

    ax_s = fig.add_subplot(gs[0, :])
    draw_schematic(ax_s, stats, compute)
    ax_s.set_title("A  Pipeline and measured timings", loc="left", fontsize=10, fontweight="bold")

    ax_w = fig.add_subplot(gs[1, :])
    n = M.shape[0]
    lo, hi = np.percentile(M, 1), np.percentile(M, 99.5)
    im = ax_w.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                     extent=[x[0], x[-1], 0, n], norm=PowerNorm(0.5, vmin=lo, vmax=hi))
    ax_w.set_xlabel("2θ (°)"); ax_w.set_ylabel("Frame (collection order)")
    if t_ok:
        idx = np.linspace(0, n - 1, min(6, n)).astype(int)
        sec = ax_w.secondary_yaxis("right")
        sec.set_yticks(idx + 0.5)
        sec.set_yticklabels([f"{t_min[k]:.0f}" for k in idx])
        sec.set_ylabel("Time (min)")
    ax_w.axvspan(peak - hw, peak + hw, color="w", alpha=0.15)
    if highlight:
        lo, hi = highlight
        idx = [k for k, c in enumerate(order) if lo <= c <= hi]
        if idx:
            ax_w.axhspan(min(idx), max(idx) + 1, color="red", alpha=0.12)
            ax_w.text(x[-1], max(idx) + 1, f"  collections {lo}-{hi}",
                     color="firebrick", fontsize=7, va="bottom", ha="left", clip_on=False)
        else:
            print(f"[info] --highlight-range {lo}-{hi} has no matched frames in this run; not drawn.")
    fig.colorbar(im, ax=ax_w, pad=0.08, label="Intensity (√ scale)")
    ax_w.set_title("B  Integrated patterns through the run", loc="left", fontsize=10, fontweight="bold")

    ax_p = fig.add_subplot(gs[2, 0])
    ax_p.plot(t_min, peak_area(x, M, peak, hw), "o-", ms=3, lw=1, color="#2c3e50")
    ax_p.set_xlabel(t_label); ax_p.set_ylabel(f"Peak area at {peak:.2f}°")
    ax_p.set_title("C  Tracked reflection", loc="left", fontsize=10, fontweight="bold")

    ax_l = fig.add_subplot(gs[2, 1])
    lat = np.array([r["latency_s"] for r in rows])
    ax_l.scatter(t_min, lat, s=10, color="#c0392b")
    if stats:
        ax_l.axhline(stats["median"], color="0.3", ls="--", lw=1)
    ax_l.set_yscale("log"); ax_l.set_xlabel(t_label); ax_l.set_ylabel("Acquired → integrated (s)")
    ax_l.set_title("D  Latency per frame", loc="left", fontsize=10, fontweight="bold")

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────
def find_calibration(calib_dir: Path, prefix: str = "X3_"):
    """Find the calibration .poni and mask files in calib_dir whose name
    starts with `prefix`. Raises with a clear message if zero or multiple
    candidates are found for either, since silently guessing wrong would
    poison the compute-time figure."""
    poni_candidates = sorted(calib_dir.glob(f"{prefix}*.poni"))
    mask_candidates = sorted(p for p in calib_dir.glob(f"{prefix}*")
                             if p.suffix.lower() in (".npy", ".h5", ".hdf5", ".edf"))

    if len(poni_candidates) != 1:
        found = ", ".join(p.name for p in poni_candidates) or "(none)"
        sys.exit(f"Expected exactly one '{prefix}*.poni' file in {calib_dir}, found: {found}. "
                 "Pass --poni explicitly instead.")
    if len(mask_candidates) != 1:
        found = ", ".join(p.name for p in mask_candidates) or "(none)"
        sys.exit(f"Expected exactly one '{prefix}*' mask file (.npy/.h5/.hdf5/.edf) in "
                 f"{calib_dir}, found: {found}. Pass --mask explicitly instead.")

    print(f"[info] calibration: poni={poni_candidates[0].name}  mask={mask_candidates[0].name}")
    return str(poni_candidates[0]), str(mask_candidates[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_folder")
    ap.add_argument("--integrated-dir")
    ap.add_argument("--out")
    ap.add_argument("--peak", type=float, help="2theta of reflection to track (default: most-changing)")
    ap.add_argument("--peak-halfwidth", type=float, default=0.10)
    ap.add_argument("--poni"); ap.add_argument("--mask")
    ap.add_argument("--calib-dir",
                    help="Folder containing the calibration files; the script finds the "
                         "single X3_*.poni and X3_* mask file in it automatically. "
                         "Use --calib-prefix to change 'X3_'. Overridden by --poni/--mask "
                         "if both of those are also given.")
    ap.add_argument("--calib-prefix", default="X3_",
                    help="Filename prefix identifying the correct calibration set "
                         "(default: X3_).")
    ap.add_argument("--npt", type=int, default=1000)
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--rmax", type=float, default=30.0)
    ap.add_argument("--frame", type=int,
                    help="Collection number to use for --poni timing (repeats it n times), "
                         "instead of the first n frames in the folder.")
    ap.add_argument("--highlight-range", nargs=2, type=int, metavar=("LO", "HI"),
                    help="Collection number range to mark on the waterfall/latency plots "
                         "as a region of interest, e.g. --highlight-range 146700 146706")
    a = ap.parse_args()

    run = Path(a.run_folder)
    out = Path(a.out) if a.out else run / "demo_figures"
    out.mkdir(parents=True, exist_ok=True)

    nxs, xy, idir = collect(run, Path(a.integrated_dir) if a.integrated_dir else None)
    rows = build_table(nxs, xy)
    print(f"{len(nxs)} .nxs, {len(xy)} integrated .xy, {len(rows)} matched  ({idir.name})")
    if not rows:
        sys.exit("Nothing to analyse.")

    for w in sanity_checks(rows):
        print(f"[WARNING] {w}")

    with open(out / "latency.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["collection", "acquired_epoch", "copied_epoch", "integrated_epoch",
                     "latency_s", "copy_latency_s"])
        for r in rows:
            wr.writerow([r["collection"], r["acq"], r["copied"], r["integrated"],
                         f"{r['latency_s']:.2f}", f"{r['copy_latency_s']:.2f}"])

    stats = summarise(rows)
    if stats:
        print(f"Latency (acquired -> integrated): median {fmt_s(stats['median'])}, "
              f"IQR {fmt_s(stats['q25'])}-{fmt_s(stats['q75'])}, 90th pct {fmt_s(stats['p90'])}, "
              f"max {fmt_s(stats['max'])}")
        print(f"  within 30 s: {stats['within_30s']:.0%} | within 60 s: {stats['within_60s']:.0%} "
              f"| within 5 min: {stats['within_5min']:.0%}")
        if "copy_median" in stats:
            print(f"  acquired -> copied (Windows creation time): median {fmt_s(stats['copy_median'])}")
        print("  Note: includes the time taken to click 'Integrate' in the Inline tab.")

    compute = None
    poni, mask = a.poni, a.mask
    if a.calib_dir and not (poni and mask):
        poni, mask = find_calibration(Path(a.calib_dir), a.calib_prefix)
    if poni:
        compute = time_integration(nxs, poni, mask, npt=a.npt, rmin=a.rmin, rmax=a.rmax,
                                   frame=a.frame)
        which = f"frame {a.frame}, repeated" if a.frame is not None else "first frames"
        print(f"Integration compute time ({which}): first call {compute['first']:.2f} s "
              f"(includes pyFAI set-up), steady-state median {compute['steady_median']:.2f} s "
              f"(n={compute['n']})")

    order = [r["collection"] for r in rows]          # already sorted by collection number
    x, M = build_stack(xy, order)
    peak = a.peak if a.peak else auto_peak(x, M, a.peak_halfwidth)
    print(f"Tracking reflection at {peak:.3f} deg "
          f"({'user-specified' if a.peak else 'auto: largest variation across run'})")

    png = out / "beamtime_demo_figure.png"
    make_figure(rows, stats, compute, x, M, order, peak, a.peak_halfwidth, png,
               highlight=tuple(a.highlight_range) if a.highlight_range else None)
    print(f"Wrote {png}\nWrote {out / 'latency.csv'}")


if __name__ == "__main__":
    main()
