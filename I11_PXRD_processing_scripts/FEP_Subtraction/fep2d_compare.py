"""
fep2d_compare.py  -  compare a 2D-corrected 1D pattern with the 1D-subtracted pattern
====================================================================================
Pure analysis (numpy / scipy / pandas / matplotlib). No Streamlit, no pyFAI.
Used by the "3 . Compare with 1D" tab in fep2d_app_v3_compare.py, and importable
from a notebook:

    import fep2d_compare as fc
    x1, y1 = fc.load_xy("1D_BakgRemoval_Run007.xy")
    x2, y2 = fc.load_xy("2D_BakgRemoval_Run007.xy")
    res = fc.compare(x1, y1, x2, y2)
    res["metrics"]; res["peaks"]; res["noise_level"]

What it does (this is exactly the Run007 comparison, 5 October 2026)
-------------------------------------------------------------------
1. Put both patterns on ONE grid. If one grid is an integer multiple finer than
   the other (Run007: 2000 vs 1000 points) the finer one is averaged in blocks, which
   lands exactly on the coarser grid. Otherwise it is boxcar-smoothed to the
   coarser step and interpolated. The method used is reported in res["notes"].
2. Scale each pattern so its own strongest peak (parabola through the top three
   points) is 1.00. All "relative" numbers below are against that peak.
3. Find peaks in both (scipy find_peaks, prominence as a fraction of the main
   peak), ignore the masked 2theta range, and match them within a tolerance.
4. Report per matched peak: position shift (2D - 1D), relative heights and their
   ratio, area ratio, and the ratio in raw counts.
5. Report correlation, noise (scatter about a running median), typical level
   (median) in chosen regions, and the curve near the mask edges.

Nothing here changes or reads the 2D frames; it only reads two (or three) .xy files.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

DEFAULTS = dict(
    mask=(5.2, 6.1),                 # 2theta range ignored when finding peaks / correlating
    prominence=0.06,                 # peak prominence, fraction of the main peak
    match_tol=0.06,                  # deg: 1D and 2D peaks closer than this are "the same"
    strong=0.20,                     # 1D relative height above which a peak is "strong"
    high_angle=17.0,                 # deg: peaks above this are summarised separately
    corr_regions=((6.5, 30.0), (1.0, 5.0)),
    noise_regions=((22.5, 30.0), (18.0, 22.0), (12.5, 14.5)),
    level_regions=((1.0, 2.0), (2.0, 4.5), (12.5, 14.5), (22.5, 30.0)),
    edge_windows=((5.30, 5.55), (5.94, 6.15)),
    median_window_deg=0.38,          # running-median window used for the noise estimate
    area_half_deg=0.12,              # peak area is integrated +/- this far from the top
)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def load_xy(src):
    """Read a two-column .xy / .dat / .csv file. Lines that are not two numbers
    (headers, a wavelength line) are skipped. `src` is a path or a bytes/str buffer.
    Returns x, y as float arrays sorted by x."""
    if isinstance(src, (str, Path)):
        text = Path(src).read_text(errors="replace")
    elif isinstance(src, bytes):
        text = src.decode(errors="replace")
    else:
        text = src.read()
        text = text.decode(errors="replace") if isinstance(text, bytes) else text
    xs, ys = [], []
    for line in text.splitlines():
        parts = re.split(r"[,\s;]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    if len(xs) < 10:
        raise ValueError("fewer than 10 numeric rows found: is this a two-column .xy file?")
    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x)
    return x[order], y[order]


def find_pairs(folder1, folder2, tag1="1D", tag2="2D", exts=(".xy", ".dat", ".csv")):
    """Match files of two folders whose names are equal once tag1 / tag2 are removed.
    '1D_BakgRemoval_Run007.xy' and '2D_BakgRemoval_Run007.xy' share the key
    'BakgRemoval_Run007'. Returns (pairs, only1, only2) with pairs = [(key, p1, p2)]."""
    def index(folder, tag):
        out = {}
        for p in sorted(Path(folder).iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                key = re.sub(r"[_\-\s]{2,}", "_", p.stem.replace(tag, "", 1)).strip("_- ")
                out[key] = p
        return out
    a, b = index(folder1, tag1), index(folder2, tag2)
    keys = sorted(set(a) & set(b))
    return ([(k, a[k], b[k]) for k in keys],
            sorted(set(a) - set(b)), sorted(set(b) - set(a)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _step(x):
    return float(np.median(np.diff(x)))


def to_common_grid(x1, y1, x2, y2):
    """Resample so both patterns share one grid. Returns xg, a (pattern 1), b (pattern 2), note."""
    s1, s2 = _step(x1), _step(x2)
    lo, hi = max(x1[0], x2[0]), min(x1[-1], x2[-1])
    if hi - lo < 5:
        raise ValueError("the two patterns overlap by less than 5 degrees 2theta")
    if abs(s1 - s2) / max(s1, s2) < 0.02:
        keep = (x1 >= lo) & (x1 <= hi)
        xg = x1[keep]
        return xg, y1[keep], np.interp(xg, x2, y2), "same grid step: second pattern interpolated onto the first"
    coarse_is_1 = s1 > s2
    xc, yc, xf, yf = (x1, y1, x2, y2) if coarse_is_1 else (x2, y2, x1, y1)
    sc, sf = max(s1, s2), min(s1, s2)
    k = sc / sf
    kr = int(round(k))
    if kr >= 2 and abs(k - kr) < 0.02:
        n = (len(xf) // kr) * kr
        xr = xf[:n].reshape(-1, kr).mean(1)
        yr = yf[:n].reshape(-1, kr).mean(1)
        # does the block-averaged fine grid sit on the coarse grid?
        probe = xr[(xr > xc[0]) & (xr < xc[-1])][:50]
        off = np.abs(probe - xc[np.abs(xc[None, :] - probe[:, None]).argmin(1)]).max() if len(probe) else 1
        if off < 0.1 * sc:
            keep = (xc >= lo) & (xc <= hi) & (xc >= xr[0]) & (xc <= xr[-1])
            xg = xc[keep]
            yc_g = yc[keep]
            yf_g = np.interp(xg, xr, yr)
            note = (f"{kr} points of the finer pattern averaged into one: both on the same "
                    f"{len(xg)}-point grid (step {sc:.4f} deg)")
        else:
            keep = (xc >= lo) & (xc <= hi)
            xg, yc_g = xc[keep], yc[keep]
            yf_g = np.interp(xg, xf, uniform_filter1d(yf, kr, mode="nearest"))
            note = f"finer pattern smoothed over {kr} points and interpolated onto the coarser grid"
    else:
        keep = (xc >= lo) & (xc <= hi)
        xg, yc_g = xc[keep], yc[keep]
        yf_g = np.interp(xg, xf, uniform_filter1d(yf, max(1, int(round(k))), mode="nearest"))
        note = f"grid steps {sf:.4f} and {sc:.4f} deg are not a whole multiple: finer pattern smoothed and interpolated"
    return (xg, yc_g, yf_g, note) if coarse_is_1 else (xg, yf_g, yc_g, note)


def _para(x, y, i, step):
    """Parabolic peak position and height from the top three points."""
    if i <= 0 or i >= len(y) - 1:
        return x[i], y[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    dx = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    dx = float(np.clip(dx, -1, 1))
    return x[i] + dx * step, y1 - 0.25 * (y0 - y2) * dx


def _area(x, y, i, half_pts):
    lo, hi = max(i - half_pts, 0), min(i + half_pts + 1, len(y))
    return float(np.trapezoid(y[lo:hi], x[lo:hi])) if hi - lo > 1 else 0.0


def _longest_zero_run(x, y, lo=4.5, hi=6.5, tol=0.02):
    """Longest stretch in [lo, hi] where the curve stays at or below tol x its own maximum
    (the flat, masked part). Returns (start, end) in 2theta."""
    m = (x >= lo) & (x <= hi)
    xs, zs = x[m], y[m] <= tol * float(np.max(y))
    best, cur_start, best_rng = 0, None, (np.nan, np.nan)
    for i, z in enumerate(zs):
        if z and cur_start is None:
            cur_start = i
        if (not z or i == len(zs) - 1) and cur_start is not None:
            end = i if z else i - 1
            if end - cur_start + 1 > best:
                best, best_rng = end - cur_start + 1, (xs[cur_start], xs[end])
            cur_start = None
    return best_rng


def parse_ranges(text):
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            lo, hi = part.split("-")
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                raise ValueError(f"range '{part}' must have high > low")
            out.append((lo, hi))
    return out


def fmt_ranges(rs):
    return ", ".join(f"{a:g}-{b:g}" for a, b in rs)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def compare(x1, y1, x2, y2, **kw):
    """Compare pattern 1 (the 1D-subtracted pattern) with pattern 2 (the 2D-corrected pattern).
    Returns a dict: metrics (flat), peaks, unmatched_1d, unmatched_2d, noise_level, edges,
    grid (x, a, b normalised and a_raw, b_raw), native (original arrays), notes, params."""
    p = {**DEFAULTS, **kw}
    xg, a_raw, b_raw, note = to_common_grid(x1, y1, x2, y2)
    step = _step(xg)
    notes = [note]
    mlo, mhi = p["mask"]
    mask = (xg >= mlo) & (xg <= mhi)

    ia, ib = int(np.argmax(a_raw)), int(np.argmax(b_raw))
    pa_pos, pa_h = _para(xg, a_raw, ia, step)
    pb_pos, pb_h = _para(xg, b_raw, ib, step)
    if pa_h <= 0 or pb_h <= 0:
        raise ValueError("a pattern has no positive peak")
    a, b = a_raw / pa_h, b_raw / pb_h
    if abs(pa_pos - pb_pos) > 0.15:
        notes.append(f"the strongest peak is at {pa_pos:.2f} deg in pattern 1 but {pb_pos:.2f} deg in "
                     "pattern 2: relative heights are against different peaks, read with care")
    if mask[ia] or mask[ib]:
        notes.append("the strongest point lies inside the masked range")

    half_pts = max(1, int(round(p["area_half_deg"] / step)))
    # ---- peaks
    fa, _ = find_peaks(a, prominence=p["prominence"])
    fb, _ = find_peaks(b, prominence=p["prominence"])
    fa = [i for i in fa if not mask[i]]
    fb = [i for i in fb if not mask[i]]
    rows, used = [], set()
    for i in fa:
        cand = [j for j in fb if abs(xg[j] - xg[i]) <= p["match_tol"] and j not in used]
        if not cand:
            continue
        j = min(cand, key=lambda j: abs(xg[j] - xg[i]))
        used.add(j)
        x_a, h_a = _para(xg, a, i, step)
        x_b, h_b = _para(xg, b, j, step)
        _, raw_a = _para(xg, a_raw, i, step)
        _, raw_b = _para(xg, b_raw, j, step)
        A1, A2 = _area(xg, a, i, half_pts), _area(xg, b, j, half_pts)
        rows.append(dict(two_theta_1D=x_a, shift_2D_minus_1D=x_b - x_a, height_1D=h_a, height_2D=h_b,
                         height_ratio=h_b / h_a if h_a > 0 else np.nan,
                         area_ratio=A2 / A1 if A1 > 0 else np.nan,
                         raw_ratio=raw_b / raw_a if raw_a > 0 else np.nan,
                         strong=bool(h_a >= p["strong"])))
    peaks = pd.DataFrame(rows)
    matched_a = {round(r["two_theta_1D"], 3) for r in rows}
    un1 = [round(float(xg[i]), 2) for i in fa if round(_para(xg, a, i, step)[0], 3) not in matched_a]
    un2 = [round(float(xg[j]), 2) for j in fb if j not in used]

    # ---- correlation
    m = {}
    for lo, hi in p["corr_regions"]:
        sel = (xg >= lo) & (xg <= hi) & ~mask
        m[f"corr_{lo:g}-{hi:g}"] = float(np.corrcoef(a[sel], b[sel])[0, 1]) if sel.sum() > 5 else np.nan

    # ---- noise and level
    W = max(3, int(round(p["median_window_deg"] / step)) | 1)
    ra = a - median_filter(a, size=W, mode="nearest")
    rb = b - median_filter(b, size=W, mode="nearest")
    nl = []
    for lo, hi in p["noise_regions"]:
        sel = (xg >= lo) & (xg <= hi) & ~mask
        if sel.sum() < 8:
            continue
        s1, s2 = float(ra[sel].std()), float(rb[sel].std())
        nl.append(dict(measure="noise (sd)", region=f"{lo:g}-{hi:g}", pattern_1D=s1, pattern_2D=s2,
                       change=f"{100 * (s2 / s1 - 1):+.0f}%" if s1 > 0 else ""))
        m[f"noise_{lo:g}-{hi:g}_1D"], m[f"noise_{lo:g}-{hi:g}_2D"] = s1, s2
        m[f"noise_{lo:g}-{hi:g}_change_pct"] = 100 * (s2 / s1 - 1) if s1 > 0 else np.nan
    for lo, hi in p["level_regions"]:
        sel = (xg >= lo) & (xg <= hi) & ~mask
        if sel.sum() < 8:
            continue
        l1, l2 = float(np.median(a[sel])), float(np.median(b[sel]))
        nl.append(dict(measure="level (median)", region=f"{lo:g}-{hi:g}", pattern_1D=l1, pattern_2D=l2,
                       change=f"{l2 - l1:+.3f}"))
        m[f"level_{lo:g}-{hi:g}_1D"], m[f"level_{lo:g}-{hi:g}_2D"] = l1, l2
        m[f"level_{lo:g}-{hi:g}_diff"] = l2 - l1
    noise_level = pd.DataFrame(nl)
    if len(p["noise_regions"]) and f"noise_{p['noise_regions'][0][0]:g}-{p['noise_regions'][0][1]:g}_1D" in m:
        lo, hi = p["noise_regions"][0]
        m["peak_over_noise_1D"] = 1 / m[f"noise_{lo:g}-{hi:g}_1D"] if m[f"noise_{lo:g}-{hi:g}_1D"] > 0 else np.nan
        m["peak_over_noise_2D"] = 1 / m[f"noise_{lo:g}-{hi:g}_2D"] if m[f"noise_{lo:g}-{hi:g}_2D"] > 0 else np.nan

    # ---- edges, on the native (unbinned) data, as a fraction of each pattern's own maximum
    edges = []
    for lo, hi in p["edge_windows"]:
        r = dict(window=f"{lo:g}-{hi:g}")
        for tag, xx, yy in (("1D", x1, y1), ("2D", x2, y2)):
            sel = (xx >= lo) & (xx <= hi)
            top = float(yy[sel].max()) if sel.any() else np.nan
            r[f"max_{tag}"] = top
            r[f"fraction_of_main_{tag}"] = top / float(yy.max()) if sel.any() and yy.max() > 0 else np.nan
            r[f"at_{tag}"] = float(xx[sel][np.argmax(yy[sel])]) if sel.any() else np.nan
        edges.append(r)
    edges = pd.DataFrame(edges)
    z1, z2 = _longest_zero_run(x1, y1), _longest_zero_run(x2, y2)

    # ---- headline numbers
    strong = peaks[peaks.strong] if len(peaks) else peaks
    m.update(
        main_peak_1D=pa_pos, main_peak_2D=pb_pos,
        raw_scale_main_2D_over_1D=float(pb_h / pa_h),
        n_peaks_1D=len(fa), n_peaks_2D=len(fb), n_matched=len(peaks), n_strong=len(strong),
        strong_median_abs_shift=float(strong.shift_2D_minus_1D.abs().median()) if len(strong) else np.nan,
        strong_max_abs_shift=float(strong.shift_2D_minus_1D.abs().max()) if len(strong) else np.nan,
        strong_median_height_ratio=float(strong.height_ratio.median()) if len(strong) else np.nan,
        strong_min_height_ratio=float(strong.height_ratio.min()) if len(strong) else np.nan,
        strong_min_height_ratio_at=float(strong.loc[strong.height_ratio.idxmin(), "two_theta_1D"]) if len(strong) else np.nan,
    )
    hi_pk = peaks[peaks.two_theta_1D > p["high_angle"]] if len(peaks) else peaks
    m["n_peaks_above_high_angle"] = len(hi_pk)
    m["high_angle_median_height_ratio"] = float(hi_pk.height_ratio.median()) if len(hi_pk) else np.nan
    for _, r in edges.iterrows():
        k = r["window"]
        m[f"edge_{k}_fraction_1D"], m[f"edge_{k}_fraction_2D"] = r["fraction_of_main_1D"], r["fraction_of_main_2D"]
    m["zero_run_1D_start"], m["zero_run_1D_end"] = z1
    m["zero_run_2D_start"], m["zero_run_2D_end"] = z2
    m["grid_step_deg"] = step
    if len(peaks) < 3:
        notes.append("fewer than 3 matched peaks: lower the prominence or check the patterns are the same frames")

    return dict(
        metrics=m, peaks=peaks, unmatched_1d=un1, unmatched_2d=un2, noise_level=noise_level, edges=edges,
        grid=dict(x=xg, a=a, b=b, a_raw=a_raw, b_raw=b_raw), native=dict(x1=x1, y1=y1, x2=x2, y2=y2),
        zero_runs=dict(one_d=z1, two_d=z2), notes=notes, params=p,
    )


# ---------------------------------------------------------------------------
# presentation helpers
# ---------------------------------------------------------------------------
def metrics_table(res):
    """The headline numbers as a short two-column table (label, value) for display."""
    m, p = res["metrics"], res["params"]
    rows = [
        ("Matched peaks (1D / 2D found)", f"{m['n_matched']}  ({m['n_peaks_1D']} / {m['n_peaks_2D']})"),
        ("Position shift, strong peaks: median / max (deg)",
         f"{m['strong_median_abs_shift']:.3f} / {m['strong_max_abs_shift']:.3f}" if m['n_strong'] else "n/a"),
    ]
    for lo, hi in p["corr_regions"]:
        rows.append((f"Correlation {lo:g}-{hi:g} deg", f"{m[f'corr_{lo:g}-{hi:g}']:.2f}"))
    rows.append(("Raw scale 2D / 1D at the main peak", f"{m['raw_scale_main_2D_over_1D']:.2f}"))
    if m["n_strong"]:
        rows.append(("Strong peaks, height 2D / 1D: median (lowest, at deg)",
                     f"{m['strong_median_height_ratio']:.2f}  ({m['strong_min_height_ratio']:.2f} at {m['strong_min_height_ratio_at']:.2f})"))
    if m["n_peaks_above_high_angle"]:
        rows.append((f"Peaks above {p['high_angle']:g} deg, median height 2D / 1D (n)",
                     f"{m['high_angle_median_height_ratio']:.2f}  ({m['n_peaks_above_high_angle']})"))
    if "peak_over_noise_1D" in m:
        lo, hi = p["noise_regions"][0]
        rows.append((f"Main peak / noise at {lo:g}-{hi:g} deg (1D, 2D)",
                     f"{m['peak_over_noise_1D']:.0f}, {m['peak_over_noise_2D']:.0f}"))
    for w in p["edge_windows"]:
        k = f"{w[0]:g}-{w[1]:g}"
        rows.append((f"Height inside {k} deg, fraction of main peak (1D, 2D)",
                     f"{m[f'edge_{k}_fraction_1D']:.2f}, {m[f'edge_{k}_fraction_2D']:.2f}"))
    rows.append(("Flat (masked) stretch: 1D (deg)", f"{m['zero_run_1D_start']:.2f} to {m['zero_run_1D_end']:.2f}"))
    rows.append(("Flat (masked) stretch: 2D (deg)", f"{m['zero_run_2D_start']:.2f} to {m['zero_run_2D_end']:.2f}"))
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def flags(res):
    """Plain statements about anything that looks off. Facts about the numbers only."""
    m, p = res["metrics"], res["params"]
    out = []
    lo, hi = p["noise_regions"][0] if p["noise_regions"] else (None, None)
    if lo is not None and f"noise_{lo:g}-{hi:g}_change_pct" in m:
        c = m[f"noise_{lo:g}-{hi:g}_change_pct"]
        out.append(f"High-angle noise ({lo:g}-{hi:g} deg) is {abs(c):.0f}% {'lower' if c < 0 else 'higher'} in 2D.")
    for lo2, hi2 in p["level_regions"]:
        d = m.get(f"level_{lo2:g}-{hi2:g}_diff")
        if d is not None and abs(d) >= 0.015:
            out.append(f"Typical level at {lo2:g}-{hi2:g} deg is {abs(d):.3f} {'higher' if d > 0 else 'lower'} in 2D "
                       "(fractions of the main peak).")
    for w in p["edge_windows"]:
        k = f"{w[0]:g}-{w[1]:g}"
        f1, f2 = m[f"edge_{k}_fraction_1D"], m[f"edge_{k}_fraction_2D"]
        if abs(f2 - f1) >= 0.08:
            out.append(f"Inside {k} deg the curve reaches {f2:.2f} of the main peak in 2D against {f1:.2f} in 1D (mask-edge feature).")
    if m["n_strong"] and m["strong_min_height_ratio"] < 0.7:
        out.append(f"The strong peak at {m['strong_min_height_ratio_at']:.2f} deg is only {m['strong_min_height_ratio']:.2f} "
                   "of its 1D height in 2D.")
    if m["n_peaks_above_high_angle"] and m["high_angle_median_height_ratio"] < 0.8:
        out.append(f"Peaks above {p['high_angle']:g} deg have a median height of {m['high_angle_median_height_ratio']:.2f} "
                   "of the 1D height in 2D.")
    if m["n_strong"] and m["strong_max_abs_shift"] > 0.05:
        out.append(f"At least one strong peak is shifted by {m['strong_max_abs_shift']:.3f} deg between 1D and 2D.")
    return out + list(res["notes"])


def summary_row(key, res):
    """One flat row for a many-runs summary table."""
    m, p = res["metrics"], res["params"]
    r = {"run": key, "matched": m["n_matched"], "shift_median": m["strong_median_abs_shift"],
         "shift_max": m["strong_max_abs_shift"]}
    for lo, hi in p["corr_regions"]:
        r[f"corr_{lo:g}-{hi:g}"] = m[f"corr_{lo:g}-{hi:g}"]
    r["raw_scale"] = m["raw_scale_main_2D_over_1D"]
    r["strong_median_ratio"] = m["strong_median_height_ratio"]
    r["strong_min_ratio"] = m["strong_min_height_ratio"]
    r["high_angle_ratio"] = m["high_angle_median_height_ratio"]
    if p["noise_regions"]:
        lo, hi = p["noise_regions"][0]
        r[f"noise_{lo:g}-{hi:g}_change_%"] = m.get(f"noise_{lo:g}-{hi:g}_change_pct")
        r["peak/noise_1D"], r["peak/noise_2D"] = m.get("peak_over_noise_1D"), m.get("peak_over_noise_2D")
    for lo, hi in p["level_regions"][:2]:
        r[f"level_{lo:g}-{hi:g}_diff"] = m.get(f"level_{lo:g}-{hi:g}_diff")
    for w in p["edge_windows"]:
        k = f"{w[0]:g}-{w[1]:g}"
        r[f"edge_{k}_1D"], r[f"edge_{k}_2D"] = m[f"edge_{k}_fraction_1D"], m[f"edge_{k}_fraction_2D"]
    return r


def make_figure(res, ref=None, label=""):
    """Three panels: overlay of both patterns (each scaled to its main peak, 2D offset),
    a zoom on the mask region, and the 2D/1D height ratio of every matched peak against 2theta."""
    import matplotlib.pyplot as plt
    g, p = res["grid"], res["params"]
    x = g["x"]
    mlo, mhi = p["mask"]
    fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, :])
    off = 1.2
    ax.plot(x, g["a"], color="#2E8B57", lw=1.0, label="1D-subtracted")
    ax.plot(x, g["b"] + off, color="#2F6DB5", lw=1.0, label=f"2D-corrected (offset {off:g})")
    if ref is not None:
        rx, ry = ref
        sel = (rx >= x[0]) & (rx <= x[-1])
        ax.plot(rx[sel], ry[sel] / ry[sel].max() * 1.0 + 2 * off, color="#7B3FA0", lw=0.9, label="reference pattern")
    ax.axvspan(mlo, mhi, color="0.85", alpha=0.7, lw=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("intensity, main peak = 1")
    ax.set_title(label or "1D vs 2D", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax2 = fig.add_subplot(gs[1, 0])
    n = res["native"]
    z = (n["x1"] > 4.8) & (n["x1"] < 7.0)
    z2 = (n["x2"] > 4.8) & (n["x2"] < 7.0)
    ax2.plot(n["x1"][z], n["y1"][z] / n["y1"].max(), color="#2E8B57", lw=1.1, marker=".", ms=3, label="1D")
    ax2.plot(n["x2"][z2], n["y2"][z2] / n["y2"].max(), color="#2F6DB5", lw=1.1, marker=".", ms=3, label="2D")
    ax2.axvspan(mlo, mhi, color="0.85", alpha=0.7, lw=0)
    for w in p["edge_windows"]:
        ax2.axvline(w[0], color="0.5", lw=0.5, ls=":")
        ax2.axvline(w[1], color="0.5", lw=0.5, ls=":")
    ax2.set_title("Mask edges (native points, each over its own maximum)", loc="left", fontsize=10)
    ax2.set_xlabel("2theta (deg)")
    ax2.set_ylabel("fraction of own maximum")
    ax2.legend(fontsize=8, frameon=False)
    ax3 = fig.add_subplot(gs[1, 1])
    pk = res["peaks"]
    if len(pk):
        s = pk[pk.strong]
        w = pk[~pk.strong]
        ax3.scatter(w.two_theta_1D, w.height_ratio, s=22, color="0.6", label="weaker peaks")
        ax3.scatter(s.two_theta_1D, s.height_ratio, s=40, color="#2F6DB5", label="strong peaks")
    ax3.axhline(1, color="0.3", lw=0.8)
    ax3.set_ylim(0, max(1.6, float(pk.height_ratio.max()) * 1.05) if len(pk) else 1.6)
    ax3.set_title("Peak height, 2D / 1D, against 2theta", loc="left", fontsize=10)
    ax3.set_xlabel("2theta (deg)")
    ax3.set_ylabel("2D height / 1D height")
    ax3.legend(fontsize=8, frameon=False)
    return fig
