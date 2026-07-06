import socket
import sys
import threading
import time
import json
import numpy as np
import os

import perception.process as perception
from perception.graph_sender import GraphSender

# ── Shared state ──────────────────────────────────────────────────────────────
_agent_pos      = None
_agent_pos_lock = threading.Lock()

# ── Config ────────────────────────────────────────────────────────────────────
SENSOR_HOST     = '127.0.0.1'
SENSOR_PORT     = 5004
GRAPH_HOST      = '127.0.0.1'
GRAPH_PORT      = 5006
UNITY_CMD_HOST  = '127.0.0.1'
UNITY_CMD_PORT  = 5008

WAYPOINT_DWELL  = 1.5

FRONTIER_VISIT_RADIUS = 1.5

MIN_FRONTIERS   = 1

IDLE_RETRY_SECS = 2.0

# ── Graph sender ──────────────────────────────────────────────────────────────
sender = GraphSender(host=GRAPH_HOST, port=GRAPH_PORT)

# ── NavMesh command sender ─────────────────────────────────────────────────────
_nav_sock      = None
_nav_sock_lock = threading.Lock()

def _send_navmesh_move(tx, tz, theta=0.0):
    """Send one move_to command to Unity over a persistent socket. Returns
    True on success. Reconnects once if the socket is broken."""
    global _nav_sock
    cmd = json.dumps({"command": "move_to", "x": tz, "y": tx, "theta": theta}) + "\n"
    with _nav_sock_lock:
        for attempt in range(2):
            if _nav_sock is None:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect((UNITY_CMD_HOST, UNITY_CMD_PORT))
                    s.settimeout(5.0)
                    _nav_sock = s
                    print(f"[NavMesh] Connected to Unity on {UNITY_CMD_HOST}:{UNITY_CMD_PORT}")
                except Exception as e:
                    print(f"[NavMesh] Could not connect: {e}")
                    _nav_sock = None
                    return False
            try:
                _nav_sock.sendall(cmd.encode("utf-8"))
                return True
            except Exception as e:
                print(f"[NavMesh] Send failed (attempt {attempt+1}): {e}")
                try: _nav_sock.close()
                except Exception: pass
                _nav_sock = None
    return False

GRAPH_SAVE_PATH = "C:\\Users\\ALIENWARE\\Unity\\Spatial_Navigation_proj\\core\\LLM\\hospital_graph.json"

# Minimum number of nodes a loaded graph must have to be considered usable.
GRAPH_MIN_NODES_FOR_READONLY = 5

# ── CLI flags ─────────────────────────────────────────────────────────────────
_FORCE_FRESH = "--fresh" in sys.argv

# ── Load previous graph if it exists ──────────────────────────────────────────
GRAPH_READ_ONLY = False
perception.GRAPH_READ_ONLY = False

if _FORCE_FRESH:
    print("[Main] --fresh flag set — ignoring any saved graph, starting fresh.")
elif os.path.exists(GRAPH_SAVE_PATH):
    try:
        loaded = perception.semantic_mapper.load(GRAPH_SAVE_PATH)
        nodes  = list(loaded.graph.nodes(data=True))
        n_nodes = len(nodes)

        if n_nodes >= GRAPH_MIN_NODES_FOR_READONLY:
            perception.semantic_mapper = loaded
            GRAPH_READ_ONLY = True
            perception.GRAPH_READ_ONLY = True
            perception.semantic_mapper.consolidate_scene_labels()

            first_pos = nodes[0][1].get('position', [0, 0, 0])
            with _agent_pos_lock:
                _agent_pos = np.array(first_pos)
            print(f"[Main] Loaded graph with {n_nodes} nodes — read-only mode. "
                  f"Agent seeded at {first_pos}.")
            print("[Main] To rebuild the graph from scratch run:  python main.py --fresh")
        else:
            print(f"[Main] Saved graph has only {n_nodes} node(s) "
                  f"(threshold={GRAPH_MIN_NODES_FOR_READONLY}) — treating as empty, "
                  f"starting fresh exploration.")
            GRAPH_READ_ONLY = False
            perception.GRAPH_READ_ONLY = False
            with _agent_pos_lock:
                _agent_pos = np.array([0, 0, 0])
    except Exception as e:
        print(f"[Main] Failed to load graph: {e} — starting fresh.")
        GRAPH_READ_ONLY = False
        perception.GRAPH_READ_ONLY = False
        with _agent_pos_lock:
            _agent_pos = np.array([0, 0, 0])
else:
    print("[Main] No saved graph found — starting fresh exploration.")
    GRAPH_READ_ONLY = False
    perception.GRAPH_READ_ONLY = False
    with _agent_pos_lock:
        _agent_pos = np.array([0, 0, 0])

# ── Navigation arbitration ────────────────────────────────────────────────────
_nav_lock       = threading.Lock()
_explore_cancel = threading.Event()
_directed_cancel = threading.Event()

LLM_CMD_HOST = '127.0.0.1'
LLM_CMD_PORT = 5010

# ── Graph persistence ──────────────────────────────────────────────────────────
GRAPH_AUTOSAVE_INTERVAL_S  = 30.0
_last_graph_save_time      = 0.0

# ── Sensor thread ─────────────────────────────────────────────────────────────
def handle_client(conn, addr):
    global _agent_pos, _last_graph_save_time
    print(f"[Sensor] Client connected from {addr}")
    buffer = ""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("[Sensor] Client closed connection")
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue

                perception.process_message(line)
                sender.send_graph(perception.semantic_mapper.graph)

                now = time.time()
                if not GRAPH_READ_ONLY and (now - _last_graph_save_time >= GRAPH_AUTOSAVE_INTERVAL_S):
                    try:
                        perception.semantic_mapper.save(GRAPH_SAVE_PATH)
                    except Exception as e:
                        print(f"[Main] Autosave failed: {e}")
                    _last_graph_save_time = now

                try:
                    msg = json.loads(line)
                    pos = np.array([msg['cam_pos_x'], msg['cam_pos_y'], msg['cam_pos_z']])
                    with _agent_pos_lock:
                        _agent_pos = pos
                    perception.occupancy_map.clear_near_agent(pos, radius_m=1.2)
                except Exception:
                    pass

    except Exception as e:
        print(f"[Sensor] Connection error: {e}")
    finally:
        conn.close()
        print("[Sensor] Client disconnected")

def sensor_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((SENSOR_HOST, SENSOR_PORT))
    srv.listen(1)
    print(f"[Sensor] Listening on {SENSOR_HOST}:{SENSOR_PORT}")
    while True:
        conn, addr = srv.accept()
        handle_client(conn, addr)

# ── Exploration thread ────────────────────────────────────────────────────────
def _pick_frontier(frontiers, agent_pos, visited):
    best_dist = float('inf')
    best      = None
    ax, az    = agent_pos[0], agent_pos[2]
    for (fx, fz) in frontiers:
        if any(np.hypot(fx - vx, fz - vz) < FRONTIER_VISIT_RADIUS
               for vx, vz in visited):
            continue
        d = np.hypot(fx - ax, fz - az)
        if d < best_dist:
            best_dist = d
            best      = (fx, fz)
    return best

def _prune_visited(visited, frontiers):
    if not frontiers:
        return visited
    pruned = []
    for vx, vz in visited:
        if any(np.hypot(fx - vx, fz - vz) < FRONTIER_VISIT_RADIUS
               for fx, fz in frontiers):
            pruned.append((vx, vz))
    return pruned

def exploration_loop():
    visited = []
    print("[Explorer] Exploration loop started — waiting for sensor data …")
    while True:
        with _agent_pos_lock:
            pos = _agent_pos
        if pos is not None:
            break
        time.sleep(0.5)
    print("[Explorer] Agent position acquired. Starting exploration.")
    while True:
        with _agent_pos_lock:
            agent_pos = _agent_pos
        if agent_pos is None:
            time.sleep(IDLE_RETRY_SECS)
            continue

        frontiers = perception.occupancy_map.get_frontiers()
        print(f"[Explorer] {len(frontiers)} frontier clusters, "
              f"{len(visited)} visited")
        visited = _prune_visited(visited, frontiers)

        if len(frontiers) < MIN_FRONTIERS:
            print("[Explorer] No frontiers — map may be fully explored. Idling …")
            time.sleep(IDLE_RETRY_SECS)
            continue

        target = _pick_frontier(frontiers, agent_pos, visited)
        if target is None:
            print("[Explorer] All current frontiers already visited. "
                  "Waiting for new ones …")
            time.sleep(IDLE_RETRY_SECS)
            continue

        tx, tz = target
        print(f"[Explorer] Navigating to frontier ({tx:.1f}, {tz:.1f})")

        with _nav_lock:
            _explore_cancel.clear()
            _directed_cancel.clear()

            ok      = _send_navmesh_move(tx, tz)
            send_ok = ok
            arrived = None
            if ok:
                arrived = _wait_for_arrival(agent_pos=_get_agent_pos,
                                            goal=(tx, tz),
                                            timeout=60.0,
                                            cancel_event=_explore_cancel)

        if not send_ok:
            print(f"[Explorer] Could not send move to ({tx:.1f}, {tz:.1f}) "
                  f"-- Unity connection issue, will retry next cycle.")
        elif arrived is None:
            print(f"[Explorer] Preempted near ({tx:.1f}, {tz:.1f}) by a "
                  f"directed command -- will reconsider next cycle.")
        elif arrived is False:
            visited.append((tx, tz))
            print(f"[Explorer] Could not path to ({tx:.1f}, {tz:.1f}), skipping.")
        else:
            visited.append((tx, tz))
            if arrived:
                print(f"[Explorer] Arrived at ({tx:.1f}, {tz:.1f}). "
                      f"Total visited: {len(visited)}")
            else:
                print(f"[Explorer] Abandoned ({tx:.1f}, {tz:.1f}). "
                      f"Total visited: {len(visited)}")
        time.sleep(0.5)

def _get_agent_pos():
    with _agent_pos_lock:
        return _agent_pos

# ── Directed navigation (only node-based) ────────────────────────────────────
def _drive_to(tx, tz):
    _directed_cancel.set()
    _explore_cancel.set()
    with _nav_lock:
        _explore_cancel.clear()
        _directed_cancel.clear()
        agent_pos = _get_agent_pos()
        if agent_pos is None:
            return "no_position", None
        ok = _send_navmesh_move(tx, tz)
        if not ok:
            return "send_failed", None
        arrived = _wait_for_arrival(
            agent_pos=_get_agent_pos,
            goal=(tx, tz),
            timeout=60.0,
            cancel_event=_directed_cancel
        )
        return "ok", arrived

def _status_from_result(result, arrived, target):
    if result == "no_position":
        return {"status": "error", "landmark": target,
                "reason": "no agent position available yet"}
    if result == "send_failed":
        return {"status": "error", "landmark": target,
                "reason": "failed to send move command to Unity (port 5008)"}
    if arrived is None:
        return {"status": "preempted", "landmark": target,
                "reason": "interrupted by a newer navigation command"}
    if arrived is False:
        return {"status": "abandoned", "landmark": target,
                "reason": "no path found"}
    return {"status": "arrived", "landmark": target}

def navigate_to_node(node_id: str):
    """
    Drive directly to a node by its ID (e.g. "lm_17").
    Called by the LLM after it has resolved the user query to a specific node.
    """
    if _get_agent_pos() is None:
        return {"status": "not_found", "landmark": None,
                "reason": "no agent position yet -- sensor feed not connected"}

    node_data = perception.semantic_mapper.graph.nodes.get(node_id)
    if node_data is None:
        return {"status": "not_found", "landmark": None,
                "reason": f"node '{node_id}' not found in graph"}

    pos = node_data.get("position", [0, 0, 0])
    tx, tz = pos[0], pos[2]
    target = {
        "id":          node_id,
        "score":       1.0,
        "position":    pos,
        "clip_label":  node_data.get("clip_label"),
        "scene_label": node_data.get("scene_label"),
        "ocr_text":    node_data.get("ocr_text", ""),
        "obs_count":   node_data.get("obs_count", 1),
    }
    print(f"[Nav] Direct node navigation to '{node_id}' "
          f"({target['clip_label']} / {target['scene_label']}) at ({tx:.1f}, {tz:.1f})")

    result, arrived = _drive_to(tx, tz)
    return _status_from_result(result, arrived, target)

def stop_navigation():
    """Cancel whatever navigation is currently in flight."""
    _explore_cancel.set()
    _directed_cancel.set()
    print("[Nav] Stop requested.")

# ── LLM command server ────────────────────────────────────────────────────────
# Only accepts navigate_to_node (and stop) – the LLM handles all query grounding.

def _send_json(conn, obj):
    try:
        conn.sendall((json.dumps(obj) + '\n').encode('utf-8'))
    except Exception as e:
        print(f"[LLMCmd] Failed to send response: {e}")

def _handle_llm_command(conn, addr):
    print(f"[LLMCmd] Client connected from {addr}")
    buffer = ""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    _send_json(conn, {"status": "error", "reason": f"bad json: {e}"})
                    continue

                cmd = msg.get("command")
                if cmd == "navigate_to_node":
                    result = navigate_to_node(msg.get("node_id", ""))
                    _send_json(conn, result)
                elif cmd == "stop":
                    stop_navigation()
                    _send_json(conn, {"status": "stopped"})
                else:
                    _send_json(conn, {"status": "error",
                                      "reason": f"unknown command '{cmd}'"})
    except Exception as e:
        print(f"[LLMCmd] Connection error: {e}")
    finally:
        conn.close()
        print("[LLMCmd] Client disconnected")

def llm_command_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LLM_CMD_HOST, LLM_CMD_PORT))
    srv.listen(5)
    print(f"[LLMCmd] Listening on {LLM_CMD_HOST}:{LLM_CMD_PORT}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=_handle_llm_command, args=(conn, addr),
                         daemon=True).start()

def _wait_for_arrival(agent_pos, goal, timeout=60.0, poll=0.25, cancel_event=None):
    STALL_WINDOW       = 8.0
    STALL_IMPROVEMENT  = 0.3
    DIVERGE_NEAR_DIST  = FRONTIER_VISIT_RADIUS * 3
    DIVERGE_WINDOW     = 12

    gx, gz   = goal
    deadline = time.time() + timeout
    stall_history  = []
    near_hist = []

    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            print(f"[Explorer] Wait-for-arrival cancelled near ({gx:.1f}, {gz:.1f}).")
            return None

        pos = agent_pos()
        if pos is not None:
            dist = np.hypot(pos[0] - gx, pos[2] - gz)
            if dist < FRONTIER_VISIT_RADIUS:
                return True

            now = time.time()
            stall_history.append((now, dist))
            stall_history = [(t, d) for t, d in stall_history
                             if now - t <= STALL_WINDOW]
            if len(stall_history) >= 4:
                oldest_dist = stall_history[0][1]
                improvement = oldest_dist - dist
                if improvement < STALL_IMPROVEMENT:
                    elapsed = now - stall_history[0][0]
                    if elapsed >= STALL_WINDOW:
                        print(f"[Explorer] Stalled near ({gx:.1f}, {gz:.1f}): "
                              f"only {improvement:.2f}m improvement in "
                              f"{elapsed:.1f}s — abandoning.")
                        return False

            if dist < DIVERGE_NEAR_DIST:
                near_hist.append(dist)
                if len(near_hist) > DIVERGE_WINDOW:
                    near_hist.pop(0)
                if len(near_hist) == DIVERGE_WINDOW:
                    if all(near_hist[i] < near_hist[i+1]
                           for i in range(DIVERGE_WINDOW - 1)):
                        print(f"[Explorer] Diverging from ({gx:.1f}, {gz:.1f}) "
                              f"while near (last={dist:.2f}m) — abandoning.")
                        return False
            else:
                near_hist.clear()

        time.sleep(poll)

    print(f"[Explorer] Arrival timeout for goal ({gx:.1f}, {gz:.1f})")
    return False

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting perception server + autonomous exploration …")

    t_sensor = threading.Thread(target=sensor_server, daemon=True)
    t_sensor.start()

    if not GRAPH_READ_ONLY:
        t_explore = threading.Thread(target=exploration_loop, daemon=True)
        t_explore.start()
        print("[Main] Exploration thread started (graph is writeable).")
    else:
        print("[Main] Graph is read‑only – exploration disabled. Only LLM commands will be processed.")

    t_llm_cmd = threading.Thread(target=llm_command_server, daemon=True)
    t_llm_cmd.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down -- saving graph...")
        try:
            if not GRAPH_READ_ONLY:
                perception.semantic_mapper.save(GRAPH_SAVE_PATH)
        except Exception as e:
            print(f"[Main] Save on shutdown failed: {e}")
        print("[Main] Shutdown complete.")