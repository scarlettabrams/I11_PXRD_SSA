#!/usr/bin/env python
"""
merge_xy.py - Merge all .xy files in a folder into a single .xy pattern.

Each .xy file (2theta, intensity) in INPUT_DIR is loaded, placed on a common
2theta grid (interpolated only if the grids differ), and combined by mean,
sum or median. The merged pattern is written to OUTPUT_DIR as
<folder_name>_merged.xy, with an optional .png overview plot.

Usage
-----
1) Edit the USER SETTINGS below and run:        python merge_xy.py
2) Or use the command line:
       python merge_xy.py "D:/Run1/04A_pre_merge_baselined" "D:/merged"
       python merge_xy.py "D:/RAW_1D" "D:/merged" --each-subfolder --method mean
   --each-subfolder merges every subfolder of INPUT_DIR separately
   (one merged .xy per run folder).
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

# ============================ USER SETTINGS ============================
INPUT_DIR = r"E:\I11BT_AP39_sept26_CBZ\Data_Processing\seeded_compiled_background_frames\seeded_compiled_background_frames_01_integrated"      # folder containing the .xy files
OUTPUT_DIR = r"E:\I11BT_AP39_sept26_CBZ\Data_Processing\seeded_compiled_background_frames\integrated_only_merge_TOPAS"  # where the merged .xy is saved
METHOD = "mean"          # "mean", "sum" or "median"
NPT = None               # grid points if interpolation is needed (None = median file length)
INCLUDE_STD = True       # add a 3rd column: std. dev. across input patterns
SAVE_PNG = True          # save an overview plot next to the merged .xy
EACH_SUBFOLDER = False   # True = merge each subfolder of INPUT_DIR separately
# =======================================================================


def read_xy(path):
    """Read a 2-column (or more) .xy/.xye file; skips headers and comments."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#!;%":
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue  # text header line
    if not rows:
        raise ValueError("no numeric data found")
    arr = np.asarray(rows)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0], arr[:, 1]


def build_common_grid(xs, npt=None):
    """Return a shared 2theta grid and whether interpolation is required."""
    x0 = xs[0]
    same = all(len(x) == len(x0) and np.allclose(x, x0, rtol=0, atol=1e-6) for x in xs)
    if same:
        return x0, False
    lo = max(x.min() for x in xs)
    hi = min(x.max() for x in xs)
    if hi <= lo:
        raise ValueError("input patterns have no overlapping 2theta range")
    n = npt or int(np.median([len(x) for x in xs]))
    full_lo = min(x.min() for x in xs)
    full_hi = max(x.max() for x in xs)
    if (hi - lo) < 0.9 * (full_hi - full_lo):
        print(f"  ! Warning: grids differ; merging only over the shared range "
              f"{lo:.3f}-{hi:.3f} deg (full range {full_lo:.3f}-{full_hi:.3f} deg)")
    return np.linspace(lo, hi, n), True


def merge_folder(input_dir, output_dir, method="mean", npt=None,
                 include_std=True, save_png=True):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = sorted(p for p in input_dir.glob("*.xy")
                   if not p.stem.endswith("_merged"))
    if not files:
        print(f"  No .xy files in {input_dir} - skipped.")
        return None

    xs, ys, used = [], [], []
    for p in files:
        try:
            x, y = read_xy(p)
            xs.append(x); ys.append(y); used.append(p)
        except Exception as e:
            print(f"  ! Skipping {p.name}: {e}")
    if not used:
        print(f"  No readable .xy files in {input_dir} - skipped.")
        return None

    grid, interpolated = build_common_grid(xs, npt)
    Y = np.vstack([np.interp(grid, x, y) if interpolated else y
                   for x, y in zip(xs, ys)])

    if method == "mean":
        merged = Y.mean(axis=0)
    elif method == "sum":
        merged = Y.sum(axis=0)
    elif method == "median":
        merged = np.median(Y, axis=0)
    else:
        raise ValueError(f"unknown method '{method}'")
    std = Y.std(axis=0, ddof=1) if len(used) > 1 else np.zeros_like(merged)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_dir.name}_merged.xy"

    header = [
        f"Merged PXRD pattern - {input_dir.name}",
        f"Created: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Source folder: {input_dir}",
        f"Method: {method} of {len(used)} patterns"
        + (" (interpolated to common grid)" if interpolated else ""),
        "Files: " + ", ".join(p.name for p in used),
        "2theta_deg  intensity" + ("  std_across_patterns" if include_std else ""),
    ]
    cols = [grid, merged, std] if include_std else [grid, merged]
    np.savetxt(out_path, np.column_stack(cols), fmt="%.6f",
               header="\n".join(header), comments="# ")
    print(f"  Merged {len(used)} files -> {out_path}")

    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        if method != "sum":
            for row in Y:
                ax.plot(grid, row, color="0.8", lw=0.5)
            ax.fill_between(grid, merged - std, merged + std,
                            color="tab:blue", alpha=0.25, label="±1σ")
        ax.plot(grid, merged, color="k", lw=1.2,
                label=f"{method} (n={len(used)})")
        ax.set_xlabel("2θ (°)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"{input_dir.name} - merged")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".png"), dpi=200)
        plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Merge all .xy files in a folder.")
    ap.add_argument("input_dir", nargs="?", default=INPUT_DIR)
    ap.add_argument("output_dir", nargs="?", default=OUTPUT_DIR)
    ap.add_argument("--method", choices=["mean", "sum", "median"], default=METHOD)
    ap.add_argument("--npt", type=int, default=NPT)
    ap.add_argument("--no-std", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--each-subfolder", action="store_true", default=EACH_SUBFOLDER)
    a = ap.parse_args()

    in_dir = Path(a.input_dir)
    if not in_dir.is_dir():
        sys.exit(f"Input folder not found: {in_dir}")

    folders = ([d for d in sorted(in_dir.iterdir()) if d.is_dir()]
               if a.each_subfolder else [in_dir])
    for folder in folders:
        print(f"Processing {folder.name} ...")
        merge_folder(folder, a.output_dir, a.method, a.npt,
                     include_std=not a.no_std, save_png=not a.no_png)


if __name__ == "__main__":
    main()
