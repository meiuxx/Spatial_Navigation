"""
graph_viz.py  —  Semantic landmark graph visualiser
=====================================================
Usage:
    python graph_viz.py                                 # looks for hospital_graph.json in cwd
    python graph_viz.py path/to/hospital_graph.json     # explicit path
    python graph_viz.py --watch path/to/...json         # auto-reload every 5 s (live exploration)

Controls (browser window):
    - Rotate / zoom / pan with mouse
    - Hover a node  → tooltip with full node details
    - Click legend  → show/hide scene label groups
    - Sidebar panel → filter by scene label, clip label, min obs_count
"""

import sys
import json
import argparse
import os
import time
import webbrowser
import threading
from pathlib import Path
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc


# ── Colour palette — one colour per scene label ───────────────────────────────

SCENE_COLORS = {
    "Administration":           "#4e79a7",
    "bathroom":                 "#59a14f",
    "conference room":          "#f28e2b",
    "corridor":                 "#e15759",
    "cafeteria":                "#76b7b2",
    "emergency":                "#ff0000",
    "examination":              "#edc948",
    "imaging":                  "#b07aa1",
    "lab":                      "#ff9da7",
    "lobby":                    "#9c755f",
    "main entrance":            "#bab0ac",
    "OR_area":                  "#d37295",
    "OR_sterilization":         "#fabfd2",
    "patient room":             "#8cd17d",
    "patient rooms hallway":    "#b6992d",
    "pharamacy":                "#499894",
    "reception-nurse station":  "#86bcb6",
    "supply room":              "#e15759",
    "VIP waiting area":         "#79706e",
    "standard waiting area":    "#d4a6c8",
    "administration lounge":    "#ccebc5",
    "unknown":                  "#aaaaaa",
    None:                       "#aaaaaa",
}


def _color(scene_label):
    return SCENE_COLORS.get(scene_label, "#aaaaaa")


def load_graph(path):
    with open(path, "r") as f:
        return json.load(f)


def build_figure(data, filter_scene=None, filter_clip=None, min_obs=1):
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # ── Apply filters ─────────────────────────────────────────────────────────
    if filter_scene:
        nodes = [n for n in nodes if n.get("scene_label") == filter_scene]
    if filter_clip:
        fc = filter_clip.lower()
        nodes = [n for n in nodes if fc in (n.get("clip_label") or "").lower()]
    nodes = [n for n in nodes if n.get("obs_count", 1) >= min_obs]

    node_ids = {n["id"] for n in nodes}

    # ── Group nodes by scene label ────────────────────────────────────────────
    by_scene = defaultdict(list)
    for n in nodes:
        by_scene[n.get("scene_label")].append(n)

    # ── Build a position lookup for edges ─────────────────────────────────────
    pos_lookup = {}
    for n in nodes:
        p = n.get("position", [0, 0, 0])
        pos_lookup[n["id"]] = p

    # ── Traces: one per scene label (so legend groups them) ──────────────────
    traces = []

    for scene_label, group in sorted(by_scene.items(), key=lambda x: str(x[0])):
        xs, ys, zs, texts, sizes, ids = [], [], [], [], [], []

        for n in group:
            p   = n.get("position", [0, 0, 0])
            obs = n.get("obs_count", 1)
            ocr = n.get("ocr_text", "") or ""
            votes_obj   = n.get("label_votes", {})
            votes_scene = n.get("scene_label_votes", {})

            # Marker size: log-scaled by obs_count so high-confidence nodes pop
            sz = 8 + 4 * np.log1p(obs)

            # Hover tooltip
            label_vote_str = ", ".join(
                f"{k}:{v}" for k, v in
                sorted(votes_obj.items(), key=lambda x: -x[1])[:4]
            ) if votes_obj else "—"
            scene_vote_str = ", ".join(
                f"{k}:{v:.2f}" for k, v in
                sorted(votes_scene.items(), key=lambda x: -x[1])[:4]
            ) if votes_scene else "—"

            tip = (
                f"<b>{n['id']}</b><br>"
                f"<b>Obj:</b> {n.get('clip_label','?')}<br>"
                f"<b>Scene:</b> {scene_label}<br>"
                f"<b>Pos:</b> ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})<br>"
                f"<b>Obs:</b> {obs}<br>"
                f"<b>OCR:</b> {ocr[:60] or '—'}<br>"
                f"<b>Obj votes:</b> {label_vote_str}<br>"
                f"<b>Scene votes:</b> {scene_vote_str}"
            )

            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            texts.append(tip)
            sizes.append(sz)
            ids.append(n["id"])

        color = _color(scene_label)
        label_name = scene_label or "unknown"

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            name=label_name,
            text=[n.get("clip_label", "?") for n in group],
            textposition="top center",
            textfont=dict(size=8, color=color),
            hovertext=texts,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.85,
                line=dict(width=0.5, color="white"),
            ),
        ))

    # ── Edge trace ────────────────────────────────────────────────────────────
    ex, ey, ez = [], [], []
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src in pos_lookup and tgt in pos_lookup:
            ps, pt = pos_lookup[src], pos_lookup[tgt]
            ex += [ps[0], pt[0], None]
            ey += [ps[1], pt[1], None]
            ez += [ps[2], pt[2], None]

    if ex:
        traces.insert(0, go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines",
            name="edges",
            line=dict(color="rgba(180,180,180,0.35)", width=1.5),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    n_nodes = len(nodes)
    n_edges = sum(1 for e in edges
                  if e.get("source") in node_ids and e.get("target") in node_ids)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"Semantic Landmark Graph — {n_nodes} nodes, {n_edges} edges"
                  + (f" | scene={filter_scene}" if filter_scene else "")
                  + (f" | clip∋'{filter_clip}'" if filter_clip else "")
                  + (f" | obs≥{min_obs}" if min_obs > 1 else "")),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(title="X (Unity)", backgroundcolor="#111", gridcolor="#333",
                       showbackground=True, zerolinecolor="#555"),
            yaxis=dict(title="Y (height)", backgroundcolor="#111", gridcolor="#333",
                       showbackground=True, zerolinecolor="#555"),
            zaxis=dict(title="Z (Unity)", backgroundcolor="#111", gridcolor="#333",
                       showbackground=True, zerolinecolor="#555"),
            bgcolor="#111111",
            aspectmode="data",
        ),
        legend=dict(
            title="Scene label",
            bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="white", size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=820,
    )

    return fig


def print_stats(data):
    nodes = data.get("nodes", [])
    print(f"\n{'='*55}")
    print(f"  Graph stats: {len(nodes)} nodes, {len(data.get('edges',[]))} edges")
    print(f"{'='*55}")

    by_scene = defaultdict(list)
    for n in nodes:
        by_scene[n.get("scene_label") or "unknown"].append(n)

    print(f"  {'Scene label':<32}  {'Nodes':>5}  {'Avg obs':>7}")
    print(f"  {'-'*46}")
    for label, group in sorted(by_scene.items(), key=lambda x: -len(x[1])):
        avg_obs = np.mean([n.get("obs_count", 1) for n in group])
        print(f"  {label:<32}  {len(group):>5}  {avg_obs:>7.1f}")

    # Detect suspiciously clustered nodes (possible confusion)
    positions = np.array([n.get("position", [0,0,0]) for n in nodes], dtype=float)
    if len(positions) > 1:
        dists = []
        for i in range(min(len(positions), 200)):
            for j in range(i+1, min(len(positions), 200)):
                dists.append(np.linalg.norm(positions[i] - positions[j]))
        median_dist = np.median(dists)
        close_pairs = sum(1 for d in dists if d < 0.5)
        print(f"\n  Median inter-node dist : {median_dist:.2f} m")
        print(f"  Pairs closer than 0.5m : {close_pairs}  ← high = possible confusion")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Semantic graph 3D visualiser")
    parser.add_argument("path", nargs="?", default="hospital_graph.json",
                        help="Path to graph JSON file")
    parser.add_argument("--watch", action="store_true",
                        help="Auto-reload every 5 seconds (for live exploration)")
    parser.add_argument("--scene", default=None,
                        help="Filter to one scene label, e.g. --scene lobby")
    parser.add_argument("--clip", default=None,
                        help="Filter clip_label substring, e.g. --clip bed")
    parser.add_argument("--min-obs", type=int, default=1,
                        help="Only show nodes with obs_count >= N (default 1)")
    parser.add_argument("--out", default=None,
                        help="Save HTML to this path instead of opening browser")
    args = parser.parse_args()

    graph_path = Path(args.path)
    if not graph_path.exists():
        # Try the two known default locations
        for candidate in [
            Path("graph_state/hospital_graph.json"),
            Path(r"C:\Users\ALIENWARE\Unity\Spatial_Navigation_proj\core\LLM\hospital_graph.json"),
        ]:
            if candidate.exists():
                graph_path = candidate
                print(f"[viz] Found graph at {graph_path}")
                break
        else:
            print(f"[viz] ERROR: could not find graph JSON at '{args.path}'")
            print("      Pass the path explicitly:  python graph_viz.py path/to/hospital_graph.json")
            sys.exit(1)

    def render(path):
        data = load_graph(path)
        print_stats(data)
        fig  = build_figure(data,
                            filter_scene=args.scene,
                            filter_clip=args.clip,
                            min_obs=args.min_obs)
        return fig

    if args.out:
        fig = render(graph_path)
        out = Path(args.out)
        fig.write_html(str(out), include_plotlyjs="cdn")
        print(f"[viz] Saved → {out}")
        return

    if args.watch:
        out_path = Path("graph_viz_live.html")
        last_mtime = 0
        print(f"[viz] Watch mode — will reload '{graph_path}' every 5 s")
        print(f"      Open {out_path.resolve()} in your browser (it will auto-refresh)")

        # Write a self-refreshing wrapper
        refresh_wrapper = out_path
        while True:
            mtime = os.path.getmtime(graph_path)
            if mtime != last_mtime:
                last_mtime = mtime
                fig = render(graph_path)
                inner_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                html = (
                    "<html><head>"
                    "<meta http-equiv='refresh' content='5'>"
                    "<style>body{margin:0;background:#1a1a1a}</style>"
                    "</head><body>"
                    + inner_html +
                    "</body></html>"
                )
                refresh_wrapper.write_text(html)
                print(f"[viz] Updated {refresh_wrapper} ({time.strftime('%H:%M:%S')})")
                if last_mtime == mtime:
                    webbrowser.open(str(refresh_wrapper.resolve()))
            time.sleep(5)
    else:
        fig = render(graph_path)
        fig.show()


if __name__ == "__main__":
    main()
