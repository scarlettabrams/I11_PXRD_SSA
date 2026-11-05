"""
live_pattern.py  -  shared helpers for the inline app and its pop-out live viewer
===================================================================================
No Streamlit in here, so the main app (fep2d_app_inline.py) and the viewer page (pages/Live_viewer.py) use exactly the same
code. Everything is read from the run folder <output folder>/<run name>/ that the worker and the app write:

    01b_sorting/ml_scores.csv           written by the worker (one row per frame, with its collection number)
    01b_sorting/sorting_decisions.csv   written by the app (your labels; they always win over the classifier's)
    02_integrated/<frame>.xy            1D pattern per frame, written by the worker (masked 2theta window bridged)
    run_events.csv                      written by the app: changes of conditions and run-finished markers
    04_merged/<run>_seg<k>_...          final pattern of a segment, written when you mark a change or the end of a run

Segments: a run is cut into segments at the markers. Segment 1 starts at the first frame; a marker at collection number N
closes the segment at N and the next one starts at N+1. The live pattern is the cumulative mean of the baselined 1D patterns
of the hit frames in the chosen segment (your label if you made one, otherwise the classifier's).
"""
from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_COLS = ["time", "collection", "event", "note", "run"]
EVENT_LABELS = {
    "change": "Conditions changed, same reference",
    "new_reference": "Conditions changed, new reference needed",
    "finished": "Run finished",
}
_CACHE: dict = {}          # (xy path, mtime, half_window, zero, ranges) -> (x, baselined y); lives as long as the process


def clean(s) -> str:
    return str(s or "").strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# settings and folders
# ---------------------------------------------------------------------------
def app_settings(app_dir) -> dict:
    """The sidebar settings the main app saved last (output folder, run name, mask, baseline settings)."""
    try:
        return json.loads((Path(app_dir) / "inline_last_session.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_dir_of(settings: dict):
    out, run = clean(settings.get("output_dir")), clean(settings.get("run_name"))
    return Path(out) / run if out and run else None


def mask_ranges(settings: dict, fp):
    if not settings.get("use_extra_mask", True):
        return []
    try:
        return fp.parse_ranges(clean(settings.get("extra_mask_text")) or fp.fmt_ranges(fp.DEFAULT_EXTRA_MASK))
    except Exception:
        return list(fp.DEFAULT_EXTRA_MASK)


# ---------------------------------------------------------------------------
# scores, decisions, labels
# ---------------------------------------------------------------------------
def read_heartbeat(run_dir):
    try:
        p = Path(run_dir) / "worker_heartbeat.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def read_scores(run_dir) -> pd.DataFrame:
    cols = ["file", "collection", "score", "ml_label", "guard"]
    try:
        p = Path(run_dir) / "01b_sorting" / "ml_scores.csv"
        if p.is_file():
            df = pd.read_csv(p).drop_duplicates("file", keep="last")
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df.sort_values("collection").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


def read_decisions(run_dir) -> dict:
    try:
        p = Path(run_dir) / "01b_sorting" / "sorting_decisions.csv"
        if p.is_file():
            d = pd.read_csv(p)
            return {str(r.file): str(r.label) for r in d.itertuples()}
    except Exception:
        pass
    return {}


def effective_labels(scores: pd.DataFrame, decisions: dict) -> dict:
    return {str(r.file): decisions.get(str(r.file), str(r.ml_label)) for r in scores.itertuples()}


# ---------------------------------------------------------------------------
# events and segments
# ---------------------------------------------------------------------------
def read_events(run_dir) -> pd.DataFrame:
    try:
        p = Path(run_dir) / "run_events.csv"
        if p.is_file():
            e = pd.read_csv(p, dtype={"note": str, "event": str, "run": str}, keep_default_na=False)
            for c in EVENT_COLS:
                if c not in e.columns:
                    e[c] = ""
            e["collection"] = pd.to_numeric(e["collection"], errors="coerce")
            return e.dropna(subset=["collection"]).sort_values(["collection", "time"], kind="stable").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=EVENT_COLS)


def append_event(run_dir, collection: int, event: str, note: str, run: str):
    if event not in EVENT_LABELS:
        raise ValueError(f"unknown event {event!r}")
    p = Path(run_dir) / "run_events.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.is_file()
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EVENT_COLS)
        if new:
            w.writeheader()
        w.writerow(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), collection=int(collection), event=event,
                        note=str(note or "").replace("\n", " ").strip(), run=run))


def segments(scores: pd.DataFrame, events: pd.DataFrame) -> list:
    """List of dicts: index (1-based), start, end (None while still open), note (what the conditions were),
    end_note, closed_by (event type or ''), open (True for the last, still growing segment)."""
    if len(scores):
        first, last = int(scores["collection"].min()), int(scores["collection"].max())
    else:
        first = last = 0
    segs, start, note, k = [], first, "start of run", 1
    for e in events.itertuples():
        end = int(e.collection)
        if end < start:                      # a marker earlier than the segment start (typed by hand): ignore it
            continue
        segs.append(dict(index=k, start=start, end=end, note=note, end_note=str(e.note or ""), closed_by=str(e.event), open=False))
        k += 1
        start = end + 1
        note = "after the run finished" if e.event == "finished" else (str(e.note or "") or "new conditions")
    segs.append(dict(index=k, start=start, end=None, note=note, end_note="", closed_by="", open=True))
    for s in segs:
        s["last"] = s["end"] if s["end"] is not None else max(last, s["start"])
    return segs


def seg_label(s: dict, n_hits=None) -> str:
    rng = f"{s['start']}-{s['end']}" if s["end"] is not None else f"{s['start']}-now"
    tail = f" · {n_hits} hits" if n_hits is not None else ""
    return f"Segment {s['index']}: {rng} · {s['note']}{tail}"


def seg_frames(scores: pd.DataFrame, labels: dict, seg: dict, include_unsure=False) -> pd.DataFrame:
    allow = {"hit", "unsure"} if include_unsure else {"hit"}
    m = scores["collection"] >= seg["start"]
    if seg["end"] is not None:
        m &= scores["collection"] <= seg["end"]
    sub = scores[m]
    return sub[[labels.get(str(f), "") in allow for f in sub["file"]]]


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------
def xy_path(run_dir, file_name) -> Path:
    return Path(run_dir) / "02_integrated" / f"{Path(str(file_name)).stem}.xy"


def baselined(run_dir, file_name, fp, half_window: int, zero: bool, ranges):
    """(x, y) of one frame's stored pattern after MOR baseline (and the masked window set to 0), cached. None if missing."""
    p = xy_path(run_dir, file_name)
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    key = (str(p), mt, int(half_window), bool(zero), tuple(map(tuple, ranges)))
    got = _CACHE.get(key)
    if got is None:
        x, y = fp.load_xy(str(p))
        yb, _ = fp.baseline_mor(x, y, int(half_window))
        if zero:
            yb = fp.zero_ranges(x, yb, ranges)
        got = (x, yb)
        if len(_CACHE) > 5000:
            _CACHE.clear()
        _CACHE[key] = got
    return got


def segment_pattern(run_dir, frames: pd.DataFrame, fp, half_window: int, zero: bool, ranges):
    """Cumulative mean of the baselined patterns of the given frames. dict(x, mean, std, n, files) or None."""
    xs, ys, used = [], [], []
    for f in frames["file"].astype(str):
        b = baselined(run_dir, f, fp, half_window, zero, ranges)
        if b is not None:
            xs.append(b[0]); ys.append(b[1]); used.append(f)
    if not ys:
        return None
    grid, mean, std = fp.merge_patterns(xs, ys)
    return dict(x=grid, mean=mean, std=std, n=len(used), files=used)


def save_segment_final(run_dir, run_name: str, seg: dict, scores, labels, fp, half_window, zero, ranges, settings, include_unsure=False, tag=""):
    """Write the segment's final merged pattern (.xy, .png) and a manifest to 04_merged. Returns the .xy path or None."""
    frames = seg_frames(scores, labels, seg, include_unsure)
    pat = segment_pattern(run_dir, frames, fp, half_window, zero, ranges)
    if pat is None:
        return None
    d = Path(run_dir) / "04_merged"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{run_name or 'run'}_seg{seg['index']}_c{seg['start']}-{seg['last']}_{'hits_unsure' if include_unsure else 'hits'}_cumulative"
    fp.save_xy(d / f"{stem}.xy", pat["x"], pat["mean"], "2theta(deg) Intensity(averaged, baselined)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pat["x"], pat["mean"], color="#1a7a4a", lw=1.0, label=f"Segment {seg['index']}: {seg['note']} ({pat['n']} hit frames)")
        ax.fill_between(pat["x"], pat["mean"] - pat["std"], pat["mean"] + pat["std"], color="#1a7a4a", alpha=0.15, label="±1σ")
        for lo, hi in ranges:
            ax.axvspan(lo, hi, color="#c0392b", alpha=0.10)
        ax.set_xlabel("2θ (°)"); ax.set_ylabel("Intensity (a.u.)"); ax.legend(); ax.grid(alpha=0.2)
        ax.set_title(f"{run_name} · {stem}")
        fig.tight_layout(); fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    except Exception:
        pass
    fp.write_manifest(d / f"{stem}_manifest.json", created=datetime.datetime.now(datetime.timezone.utc).isoformat(), run=run_name,
                      segment=seg["index"], first_collection=seg["start"], last_collection=seg["last"], note=seg["note"],
                      closed_by=seg.get("closed_by"), end_note=seg.get("end_note"), n_frames=pat["n"], frames_used=pat["files"],
                      labels="your label if you made one, otherwise the ML label", included=("hit + unsure" if include_unsure else "hit only"),
                      half_window=int(half_window), masked_window_zeroed_after_baseline=bool(zero), extra_mask_2theta_deg=[list(r) for r in ranges],
                      mean_of="per-frame baselined 1D patterns", trigger=tag, app_settings={k: settings.get(k) for k in ("npt", "rmin", "rmax")})
    return d / f"{stem}.xy"


# ---------------------------------------------------------------------------
# reference patterns for the overlay
# ---------------------------------------------------------------------------
def list_reference_files(folder) -> list:
    try:
        return sorted(p for p in Path(clean(folder)).iterdir() if p.suffix.lower() in (".xy", ".xye", ".dat", ".txt", ".csv"))
    except Exception:
        return []


def load_reference(path):
    """Two-column pattern (2theta, intensity) from a text file with any header lines. (x, y) or None."""
    try:
        df = pd.read_csv(path, sep=r"[\s,;]+", comment="#", header=None, engine="python", usecols=[0, 1], on_bad_lines="skip")
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 5:
            return None
        return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    except Exception:
        return None
