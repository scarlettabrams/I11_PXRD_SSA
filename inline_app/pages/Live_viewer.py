"""
Live_viewer.py  -  live cumulative 1D PXRD pattern, in its own browser window
==============================================================================
Streamlit multipage: this file lives in the 'pages' folder next to fep2d_app_inline.py, and opens at  <app address>/Live_viewer
(the main app's sidebar has a button that opens it in a new window, so you can park it on a second screen).

It only READS files the worker and the main app write, so it never slows the main app down:
    01b_sorting/ml_scores.csv, sorting_decisions.csv, 02_integrated/*.xy, run_events.csv
The pattern shown is the cumulative mean of the baselined 1D patterns of the HIT frames in the chosen segment.
Hit = your label if you made one, otherwise the classifier's. The classifier was trained on real diffraction data.
"""
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
import fep2d_pipeline as fp  # noqa: E402
import live_pattern as lp    # noqa: E402

st.set_page_config(page_title="Live PXRD pattern", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:0.5rem}</style>", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("lv_refresh", 10)
ss.setdefault("lv_pause", False)
ss.setdefault("lv_unsure", False)
ss.setdefault("lv_latest", True)
ss.setdefault("lv_ref_folder", "")
ss.setdefault("lv_refs", [])
ss.setdefault("lv_ref_gap", 0.15)
ss.setdefault("lv_prev", True)

settings = lp.app_settings(APP_DIR)
if not ss.lv_ref_folder:
    ss.lv_ref_folder = lp.clean(settings.get("ref_folder_viewer", "")) or lp.clean(settings.get("view_ref_path", ""))

with st.sidebar:
    st.header("Live viewer")
    st.number_input("Refresh every (s)", 2, 120, key="lv_refresh", step=1)
    st.checkbox("Pause updating", key="lv_pause", help="Freezes the picture so you can zoom and read it. Untick to carry on.")
    st.checkbox("Also include unsure frames", key="lv_unsure")
    st.checkbox("Show the newest single frame (faint)", key="lv_latest")
    st.checkbox("Show earlier finished segments (below)", key="lv_prev")
    st.divider()
    st.subheader("Reference patterns")
    st.text_input("Folder of reference patterns (.xy .xye .dat .txt .csv)", key="lv_ref_folder",
                  placeholder=r"E:/AP40/reference_patterns")
    files = lp.list_reference_files(ss.lv_ref_folder)
    names = [p.name for p in files]
    ss.lv_refs = [n for n in ss.lv_refs if n in names]
    st.multiselect("Show these", names, key="lv_refs")
    st.slider("Gap between stacked patterns", 0.0, 1.0, key="lv_ref_gap", step=0.05)
    st.caption("Reference patterns are scaled to the height of the live pattern and stacked underneath it. Only for comparing peak positions.")

st.title("Live PXRD pattern")


def _draw():
    import plotly.graph_objects as go
    s = lp.app_settings(APP_DIR)
    rd = lp.run_dir_of(s)
    if rd is None or not rd.is_dir():
        st.info("No run folder yet. Set the output folder and run name in the main app and start the worker.")
        return
    scores = lp.read_scores(rd)
    if scores.empty:
        st.info(f"Waiting for the first scored frame in `{rd}`.")
        return
    labels = lp.effective_labels(scores, lp.read_decisions(rd))
    events = lp.read_events(rd)
    segs = lp.segments(scores, events)
    hw = int(s.get("half_window", 4))
    zero = bool(s.get("zero_masked", True))
    ranges = lp.mask_ranges(s, fp)

    counts = {k: sum(1 for v in labels.values() if v == k) for k in ("hit", "non_hit", "unsure", "bad")}
    opts = list(range(len(segs)))
    fmt = lambda i: lp.seg_label(segs[i], len(lp.seg_frames(scores, labels, segs[i], bool(ss.lv_unsure))))
    if "lv_seg" not in ss or ss.lv_seg not in opts:
        ss.lv_seg = opts[-1]
    c1, c2 = st.columns([3, 2])
    c1.selectbox("Segment", opts, format_func=fmt, key="lv_seg")
    seg = segs[ss.lv_seg]
    frames = lp.seg_frames(scores, labels, seg, bool(ss.lv_unsure))
    pat = lp.segment_pattern(rd, frames, fp, hw, zero, ranges)

    hb = lp.read_heartbeat(rd) if hasattr(lp, "read_heartbeat") else None
    c2.markdown(f"**Run** `{rd.name}` · hits **{counts['hit']}** · non-hit {counts['non_hit']} · unsure {counts['unsure']} · "
                f"updated {time.strftime('%H:%M:%S')}" + (" · ⏸ paused" if ss.lv_pause else ""))
    if pat is None:
        st.info("No hit frames with a stored 1D pattern in this segment yet.")
        return

    fig = go.Figure()
    top = float(np.nanmax(pat["mean"])) or 1.0
    for lo, hi in ranges:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#c0392b", opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=pat["x"], y=pat["mean"], mode="lines", name=f"cumulative mean ({pat['n']} frames)",
                             line=dict(color="#1a7a4a", width=1.6)))
    if ss.lv_latest:
        b = lp.baselined(rd, frames["file"].iloc[-1], fp, hw, zero, ranges)
        if b is not None:
            fig.add_trace(go.Scatter(x=b[0], y=b[1], mode="lines", name=f"newest hit ({Path(str(frames['file'].iloc[-1])).stem})",
                                     line=dict(color="#7f8c8d", width=0.8), opacity=0.5))
    gap = top * float(ss.lv_ref_gap)
    level = -gap
    if ss.lv_prev:
        for sg in segs:
            if sg["index"] == seg["index"] or sg["end"] is None:
                continue
            p2 = lp.segment_pattern(rd, lp.seg_frames(scores, labels, sg, bool(ss.lv_unsure)), fp, hw, zero, ranges)
            if p2 is None:
                continue
            m = float(np.nanmax(p2["mean"])) or 1.0
            fig.add_trace(go.Scatter(x=p2["x"], y=p2["mean"] / m * top + level - top, mode="lines",
                                     name=f"segment {sg['index']}: {sg['note']}", line=dict(width=1.1)))
            level -= top + gap
    folder = Path(lp.clean(ss.lv_ref_folder))
    for n in ss.lv_refs:
        r = lp.load_reference(folder / n)
        if r is None:
            continue
        x, y = r
        y = (y - np.nanmin(y)) / ((np.nanmax(y) - np.nanmin(y)) or 1.0)
        fig.add_trace(go.Scatter(x=x, y=y * top + level - top, mode="lines", name=f"ref: {n}", line=dict(width=1.1, dash="dot")))
        level -= top + gap
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), uirevision="keep-zoom",
                      xaxis_title="2θ (°)", yaxis_title="Intensity (a.u., baselined)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True, key="lv_chart")
    if len(events):
        st.caption("Markers so far: " + " · ".join(f"{int(e.collection)} {lp.EVENT_LABELS.get(e.event, e.event)}"
                                                 + (f" ({e.note})" if str(e.note) not in ("", "nan") else "") for e in events.itertuples()))


_frag = getattr(st, "fragment", None)
if _frag is None:
    _draw()
    st.caption("Streamlit 1.37 or newer is needed for automatic refresh; press R to refresh by hand.")
else:
    _every = None if ss.lv_pause else float(ss.lv_refresh)
    _frag(run_every=_every)(_draw)()
