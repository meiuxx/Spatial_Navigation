import socket
import threading
import time
import numpy as np

import perception.process as perception
from perception.graph_sender import GraphSender
from navigation.astar_planner import AStarPlanner

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


# ── Sensor thread ─────────────────────────────────────────────────────────────

def handle_client(conn, addr):
    global _agent_pos
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

                try:
                    import json
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
    """
    Choose the closest unvisited frontier centroid to the agent.
    Returns (wx, wz) or None.
    """
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
    """
    FIX (Bug 3): Remove visited entries that no longer have any current frontier
    centroid within FRONTIER_VISIT_RADIUS.  This prevents the visited list from
    permanently blacklisting map regions that are later re-discovered as new
    frontiers at a slightly different position.

    Without this, as frontiers shift slightly with map updates, all of them
    eventually fall within FRONTIER_VISIT_RADIUS of an old visited point and
    the agent stops navigating entirely.
    """
    if not frontiers:
        return visited  # keep all if no frontiers yet; don't wipe history

    pruned = []
    for vx, vz in visited:
        # Keep this visited entry only if at least one current frontier is
        # still close to it — i.e. it's still "covering" a real frontier.
        if any(np.hypot(fx - vx, fz - vz) < FRONTIER_VISIT_RADIUS
               for fx, fz in frontiers):
            pruned.append((vx, vz))
    return pruned


def exploration_loop():
    """
    Continuously:
      1. Wait until the map has enough frontier clusters.
      2. Pick the nearest unvisited one.
      3. Plan an A* path and send waypoints to Unity.
      4. Mark it visited once the agent is close enough.
      5. Prune the visited list against current frontiers so stale entries
         don't permanently block navigation.
      6. Repeat until no frontiers remain.
    """
    planner = AStarPlanner(
        occ_map          = perception.occupancy_map,
        unity_host       = UNITY_CMD_HOST,
        unity_port       = UNITY_CMD_PORT,
        inflation_radius = 0.4,
        waypoint_spacing = 10,
        send_timeout     = WAYPOINT_DWELL,
    )

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

        # FIX (Bug 3): prune stale visited entries each cycle so old blacklist
        # entries don't permanently suppress re-discovered frontiers.
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

        success = planner.navigate_to(agent_pos, tx, tz)

        if success:
            arrived = _wait_for_arrival(agent_pos=lambda: _get_agent_pos(),
                                        goal=(tx, tz),
                                        timeout=60.0)
            visited.append((tx, tz))
            if arrived:
                print(f"[Explorer] Arrived at ({tx:.1f}, {tz:.1f}). "
                      f"Total visited: {len(visited)}")
            else:
                print(f"[Explorer] Abandoned ({tx:.1f}, {tz:.1f}). "
                      f"Total visited: {len(visited)}")
        else:
            visited.append((tx, tz))
            print(f"[Explorer] Could not path to ({tx:.1f}, {tz:.1f}), skipping.")

        time.sleep(0.5)


def _get_agent_pos():
    with _agent_pos_lock:
        return _agent_pos


def _wait_for_arrival(agent_pos, goal, timeout=60.0, poll=0.25):
    """
    Block until the agent is within FRONTIER_VISIT_RADIUS of goal.

    FIX (Bug 1): The original divergence check fired after just 8 consecutive
    rising distance samples (2 seconds).  Unity's CommandReceiver rotates the
    agent before translating, which *always* increases distance briefly at the
    start of each waypoint — causing a false-positive bail-out on nearly every
    frontier.

    The fix uses two separate guards instead of a single strict-monotone check:

    1. STALL guard  — if the agent hasn't closed distance by more than
       STALL_IMPROVEMENT metres over the last STALL_WINDOW seconds, it's
       probably stuck.  This correctly ignores the brief rotational phase.

    2. DIVERGE guard — if the agent is moving *away* AND is already very close
       to the goal, something structural is wrong (e.g. nav-mesh pushed it past
       the goal).  Only fires once close enough that we know the agent reached
       the vicinity.

    Returns True if arrived, False if abandoned.
    """
    STALL_WINDOW       = 8.0   # seconds: window to measure progress
    STALL_IMPROVEMENT  = 0.3   # metres: minimum progress required over window
    DIVERGE_NEAR_DIST  = FRONTIER_VISIT_RADIUS * 3  # only check diverge when this close
    DIVERGE_WINDOW     = 12    # consecutive rising samples (3 seconds) while near goal

    gx, gz   = goal
    deadline = time.time() + timeout

    # Ring buffer for stall detection: (timestamp, distance) pairs
    stall_history  = []
    # Ring buffer for near-goal divergence detection
    near_hist = []

    while time.time() < deadline:
        pos = agent_pos()
        if pos is not None:
            dist = np.hypot(pos[0] - gx, pos[2] - gz)

            # ── Arrival check ─────────────────────────────────────────────
            if dist < FRONTIER_VISIT_RADIUS:
                return True

            now = time.time()
            stall_history.append((now, dist))

            # ── Stall guard ───────────────────────────────────────────────
            # Drop samples older than STALL_WINDOW
            stall_history = [(t, d) for t, d in stall_history
                             if now - t <= STALL_WINDOW]
            if len(stall_history) >= 4:
                oldest_dist = stall_history[0][1]
                improvement = oldest_dist - dist   # positive = getting closer
                if improvement < STALL_IMPROVEMENT:
                    elapsed = now - stall_history[0][0]
                    if elapsed >= STALL_WINDOW:
                        print(f"[Explorer] Stalled near ({gx:.1f}, {gz:.1f}): "
                              f"only {improvement:.2f}m improvement in "
                              f"{elapsed:.1f}s — abandoning.")
                        return False

            # ── Near-goal diverge guard ───────────────────────────────────
            # Only check this when the agent is already close, so the normal
            # rotation-then-translate startup phase doesn't false-trigger it.
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
                # Reset near-history if agent wanders back out of the zone
                near_hist.clear()

        time.sleep(poll)

    print(f"[Explorer] Arrival timeout for goal ({gx:.1f}, {gz:.1f})")
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting perception server + autonomous exploration …")

    t_sensor = threading.Thread(target=sensor_server, daemon=True)
    t_sensor.start()

    t_explore = threading.Thread(target=exploration_loop, daemon=True)
    t_explore.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")