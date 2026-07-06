#!/usr/bin/env python3
# llm.py - hospital navigation agent
# Graph is cached in memory and reloaded only when the file changes.
# The LLM only sees the top TOP_K candidates by CLIP similarity, with
# embeddings stripped out before they're added to the prompt.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import socket
import sys
import threading
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from perception.clipy import CLIPModel
import ollama

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "llama3.1:8b"
MAIN_PY_HOST    = "127.0.0.1"
MAIN_PY_PORT    = 5010
GRAPH_SAVE_PATH = "hospital_graph.json"
UNITY_LLM_HOST  = "127.0.0.1"
UNITY_LLM_PORT  = 5012

TOP_K           = 15     # nodes sent to the LLM
LLM_RETRIES     = 2

# ── Graph cache ────────────────────────────────────────────────────────────────
_graph_cache: dict | None = None
_graph_mtime: float = 0.0


def _load_graph_if_stale(path: str) -> None:
    """Reload graph from disk only when the file has changed."""
    global _graph_cache, _graph_mtime

    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        _graph_cache = None
        return

    if _graph_cache is not None and mtime <= _graph_mtime:
        return  # still fresh

    with open(path) as f:
        raw = json.load(f)

    embeddings: dict[str, np.ndarray] = {}
    slim_nodes: list[dict] = []
    id_to_node: dict[str, dict] = {}

    for node in raw.get("nodes", []):
        nid   = node["id"]
        embed = np.array(node.get("clip_embed", []), dtype=np.float32)

        if embed.size > 0:
            norm = np.linalg.norm(embed)
            embeddings[nid] = embed / (norm + 1e-8)

        slim = {k: v for k, v in node.items() if k != "clip_embed"}
        slim_nodes.append(slim)
        id_to_node[nid] = slim

    adj: dict[str, list[str]] = defaultdict(list)
    for edge in raw.get("edges", []):
        src, tgt = edge.get("source"), edge.get("target")
        if src and tgt:
            adj[src].append(tgt)
            adj[tgt].append(src)

    _graph_cache = {
        "nodes":      slim_nodes,
        "embeddings": embeddings,
        "adj":        dict(adj),
        "id_to_node": id_to_node,
    }
    _graph_mtime = mtime
    print(f"[Graph] Loaded {len(slim_nodes)} nodes from {path}")


# ── CLIP (lazy singleton) ──────────────────────────────────────────────────────
_clip_model = None

def get_clip() -> "CLIPModel":
    global _clip_model
    if _clip_model is None:
        _clip_model = CLIPModel()
    return _clip_model


def clip_rank(query: str, embeddings: dict[str, np.ndarray]) -> list[tuple[float, str]]:
    """Return [(sim, node_id), ...] sorted descending."""
    if not embeddings:
        return []
    clip = get_clip()
    vec  = clip.encode_text([query])[0].cpu().detach().numpy().astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8

    scores = [(float(np.dot(vec, emb)), nid) for nid, emb in embeddings.items()]
    scores.sort(reverse=True)
    return scores


# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM = """\
You are the navigation AI of a hospital guide robot.
Pick the single best landmark for the visitor's query.

Rules:
- Read ALL candidates before deciding.
- OCR text on a node (e.g. a door sign) is the strongest signal.
- Prefer nodes with high obs count and "high-conf" tag.
- Use synonyms: reception/front-desk/nurse-station, toilet/WC/restroom, etc.
- Neighbours give spatial context (e.g. nearby OR nodes → surgical zone).
- If nothing matches, return best_id=null.

Reply with ONLY valid JSON, no markdown:
{"reasoning":"<≤80 words>","best_id":"<id or null>","confidence":"<high|medium|low>","reason":"<one sentence for visitor>"}"""


def _conf(obs: int, sal: float) -> str:
    if obs >= 5 and sal >= 0.15: return "high-conf"
    if obs >= 2 or sal >= 0.12:  return "med-conf"
    return "low-conf"


def _votes(d: dict) -> str:
    if not d: return ""
    top = sorted(d.items(), key=lambda x: -x[1])[:2]
    return "/".join(f"{k}×{v:.1f}" if isinstance(v, float) else f"{k}×{v}" for k, v in top)


def _node_line(node: dict, sim: float, adj: dict, id_to_node: dict) -> str:
    nid   = node["id"]
    pos   = node.get("position", [0,0,0])
    label = node.get("clip_label") or "?"
    scene = node.get("scene_label") or "?"
    ocr   = (node.get("ocr_text") or "").strip()
    obs   = node.get("obs_count", 1)
    sal   = node.get("best_saliency") or node.get("saliency_mean") or 0.0
    lv    = _votes(node.get("label_votes", {}))
    sv    = _votes(node.get("scene_label_votes", {}))
    nb_labels = [
        (id_to_node.get(nb, {}).get("clip_label") or id_to_node.get(nb, {}).get("scene_label") or nb)
        for nb in adj.get(nid, [])[:3]
    ]

    parts = [
        f"{nid}",
        f"({pos[0]:.1f},{pos[2]:.1f})",
        f"label={label!r}",
        f"scene={scene!r}",
    ]
    if ocr:          parts.append(f"ocr={ocr!r}")
    if lv:           parts.append(f"lvotes={lv}")
    if sv:           parts.append(f"svotes={sv}")
    parts.append(f"obs={obs}")
    parts.append(_conf(obs, sal))
    parts.append(f"clip={sim:.2f}")
    if nb_labels:    parts.append(f"near=[{','.join(nb_labels)}]")
    return "• " + " | ".join(parts)


def build_prompt(query: str, candidates: list[tuple[float, str]],
                 id_to_node: dict, adj: dict) -> str:
    lines = [SYSTEM, "", f'Query: "{query}"', "", "Candidates:"]
    for sim, nid in candidates:
        node = id_to_node.get(nid)
        if node:
            lines.append(_node_line(node, sim, adj, id_to_node))
    lines += ["", "Respond with the JSON object now."]
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────

def ground(query: str, candidates: list[tuple[float, str]],
           id_to_node: dict, adj: dict,
           model: str) -> tuple[str | None, str]:
    valid = {nid for _, nid in candidates}
    prompt = build_prompt(query, candidates, id_to_node, adj)

    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw = resp["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()

            data       = json.loads(raw)
            best_id    = data.get("best_id")
            reason     = data.get("reason", "")
            confidence = data.get("confidence", "?")
            reasoning  = data.get("reasoning", "")

            print(f"[LLM] {reasoning}")
            print(f"[LLM] → {best_id} ({confidence}): {reason}")

            if best_id is None:
                return None, reason
            if best_id in valid:
                return best_id, reason

            print(f"[LLM] Invalid id {best_id!r}, retrying…")

        except json.JSONDecodeError as e:
            print(f"[LLM] JSON error attempt {attempt+1}: {e} | raw={raw[:200]}")
        except Exception as e:
            print(f"[LLM] Error attempt {attempt+1}: {e}")

        if attempt < LLM_RETRIES:
            print("[LLM] Retrying…")

    return None, "LLM failed after retries."


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_query(user_query: str, model: str = OLLAMA_MODEL) -> str:
    t0 = time.time()
    _load_graph_if_stale(GRAPH_SAVE_PATH)
    if not _graph_cache or not _graph_cache["nodes"]:
        return "✗ No graph found. Run exploration first."

    embeddings = _graph_cache["embeddings"]
    id_to_node = _graph_cache["id_to_node"]
    adj        = _graph_cache["adj"]

    ranked     = clip_rank(user_query, embeddings)   # [(sim, id), ...]
    candidates = ranked[:TOP_K]
    t1 = time.time()

    if not candidates:
        return "✗ No landmarks with embeddings found."

    print(f"[LLM] Top CLIP: {candidates[0][1]} ({candidates[0][0]:.3f})  [{t1-t0:.1f}s]")

    best_id, reason = ground(user_query, candidates, id_to_node, adj, model)
    t2 = time.time()
    print(f"[Timing] CLIP rank {t1-t0:.1f}s | LLM ground {t2-t1:.1f}s")

    # Fallback: top CLIP match
    if best_id is None:
        best_id = candidates[0][1]
        print(f"[LLM] Falling back to top CLIP node: {best_id}")

    cmd    = {"command": "navigate_to_node", "node_id": best_id}
    result = send_command(cmd)
    t3 = time.time()
    print(f"[Timing] nav+drive {t3-t2:.1f}s | TOTAL {t3-t0:.1f}s")
    return format_result(result, cmd, reason)


# ── Nav command sender ────────────────────────────────────────────────────────

def send_command(cmd: dict, timeout: float = 90.0) -> dict:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((MAIN_PY_HOST, MAIN_PY_PORT))
            s.settimeout(timeout)
            s.sendall((json.dumps(cmd) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk: break
                buf += chunk
        return json.loads(buf.decode().strip())
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def format_result(result: dict, cmd: dict, llm_reason: str = "") -> str:
    status = result.get("status", "unknown")
    lm     = result.get("landmark") or {}
    label  = lm.get("clip_label") or lm.get("label") or cmd.get("node_id") or ""
    reason = result.get("reason", "")
    suffix = f" ({llm_reason})" if llm_reason else ""
    if status == "arrived":   return f"✓ Arrived at: {label}{suffix}"
    if status == "abandoned": return f"⚠ Could not reach. {reason}"
    if status == "not_found": return f"✗ Not in map. {reason}"
    if status == "preempted": return f"↩ Interrupted. {reason}"
    if status == "stopped":   return "■ Stopped."
    if status == "error":     return f"✗ Error: {reason}"
    return f"Reply: {result}"


# ── Unity server ───────────────────────────────────────────────────────────────

def _handle_unity(conn, addr):
    print(f"[UnityLLM] Connection from {addr}")
    try:
        conn.settimeout(5.0)
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(1024)
            if not chunk: break
            buf += chunk
        query = buf.decode().strip()
        if not query:
            conn.sendall(b"? Empty query\n"); return
        print(f"[UnityLLM] Query: {query!r}")
        conn.settimeout(None)
        result = run_query(query)
        conn.sendall((result + "\n").encode())
    except Exception as e:
        try: conn.sendall(f"✗ Error: {e}\n".encode())
        except Exception: pass
    finally:
        conn.close()


def unity_llm_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((UNITY_LLM_HOST, UNITY_LLM_PORT))
    srv.listen(5)
    print(f"[UnityLLM] Listening on {UNITY_LLM_HOST}:{UNITY_LLM_PORT}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=_handle_unity, args=(conn, addr), daemon=True).start()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    global OLLAMA_MODEL, MAIN_PY_PORT, GRAPH_SAVE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--port",  type=int, default=MAIN_PY_PORT)
    parser.add_argument("--graph", type=str, default=GRAPH_SAVE_PATH)
    args = parser.parse_args()
    OLLAMA_MODEL    = args.model
    MAIN_PY_PORT    = args.port
    GRAPH_SAVE_PATH = args.graph

    if args.query:
        print(run_query(args.query)); return

    print("Warming up CLIP + Ollama before accepting connections…")
    get_clip()  # force the CLIP load now instead of on the first query
    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            options={"temperature": 0.0},
        )
        print(f"[Warmup] {OLLAMA_MODEL} loaded and ready.")
    except Exception as e:
        print(f"[Warmup] Ollama warm-up failed (will still try per-query): {e}")

    threading.Thread(target=unity_llm_server, daemon=True).start()
    print(f"Navigator — model={OLLAMA_MODEL}  graph={GRAPH_SAVE_PATH}")

    while True:
        try:
            inp = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not inp: continue
        if inp.lower() in {"quit","exit","q"}: break
        print(f"Agent: {run_query(inp)}\n")


if __name__ == "__main__":
    main()