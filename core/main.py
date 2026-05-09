import socket
import threading
import time
import numpy as np

import perception.process as perception
from perception.graph_sender import GraphSender
from navigation.astar_planner import AStarPlanner

# ── Shared state ──────────────────────────────────────────────────────────────
# The latest known agent position, written by the sensor thread,
# read by the exploration thread.
_agent_pos      = None
_agent_pos_lock = threading.Lock()

# ── Config ────────────────────────────────────────────────────────────────────
SENSOR_HOST     = '127.0.0.1'
SENSOR_PORT     = 5004          # Unity → Python  (SensorSender.cs)
GRAPH_HOST      = '127.0.0.1'
GRAPH_PORT      = 5006          # Python → graph viewer
UNITY_CMD_HOST  = '127.0.0.1'
UNITY_CMD_PORT  = 5008          # Python → Unity  (CommandReceiver.cs)

# How long to wait (seconds) at each waypoint before sending the next one.
# Tune to match your agent's movement speed.
WAYPOINT_DWELL  = 1.5

# How close (metres) the agent must be to a frontier centroid before we
# consider it "visited" and skip re-navigating to it.
FRONTIER_VISIT_RADIUS = 1.5

# Minimum number of frontier clusters required before we attempt navigation.
MIN_FRONTIERS   = 1

# Seconds to wait between exploration ticks when no frontier is reachable.
IDLE_RETRY_SECS = 2.0

# ── Graph sender ──────────────────────────────────────────────────────────────
sender = GraphSender(host=GRAPH_HOST, port=GRAPH_PORT)


# ── Sensor thread (unchanged logic, now also updates _agent_pos) ──────────────

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

                # Keep latest agent position for the exploration thread.
                # process_message already parsed cam_pos; re-parse cheaply.
                try:
                    import json
                    msg = json.loads(line)
                    pos = np.array([msg['cam_pos_x'], msg['cam_pos_y'], msg['cam_pos_z']])
                    with _agent_pos_lock:
                        _agent_pos = pos

                    # Clear stale frontiers right around the agent every frame.
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
        # Skip if we've already been close to this one
        if any(np.hypot(fx - vx, fz - vz) < FRONTIER_VISIT_RADIUS
               for vx, vz in visited):
            continue
        d = np.hypot(fx - ax, fz - az)
        if d < best_dist:
            best_dist = d
            best      = (fx, fz)
    return best


def exploration_loop():
    """
    Continuously:
      1. Wait until the map has enough frontier clusters.
      2. Pick the nearest unvisited one.
      3. Plan an A* path and send waypoints to Unity.
      4. Mark it visited once the agent is close enough.
      5. Repeat until no frontiers remain.
    """
    planner = AStarPlanner(
        occ_map          = perception.occupancy_map,
        unity_host       = UNITY_CMD_HOST,
        unity_port       = UNITY_CMD_PORT,
        inflation_radius = 0.4,
        waypoint_spacing = 10,
        send_timeout     = WAYPOINT_DWELL,
    )

    visited = []   # list of (wx, wz) centroids we've already navigated to

    print("[Explorer] Exploration loop started — waiting for sensor data …")

    # Wait until we have a first agent position.
    while True:
        with _agent_pos_lock:
            pos = _agent_pos
        if pos is not None:
            break
        time.sleep(0.5)

    print("[Explorer] Agent position acquired. Starting exploration.")

    while True:
        # ── Snapshot agent position ──────────────────────────────────────────
        with _agent_pos_lock:
            agent_pos = _agent_pos

        if agent_pos is None:
            time.sleep(IDLE_RETRY_SECS)
            continue

        # ── Get current frontier clusters ────────────────────────────────────
        frontiers = perception.occupancy_map.get_frontiers()
        print(f"[Explorer] {len(frontiers)} frontier clusters, "
              f"{len(visited)} visited")

        if len(frontiers) < MIN_FRONTIERS:
            print("[Explorer] No frontiers — map may be fully explored. Idling …")
            time.sleep(IDLE_RETRY_SECS)
            continue

        # ── Pick target ──────────────────────────────────────────────────────
        target = _pick_frontier(frontiers, agent_pos, visited)
        if target is None:
            print("[Explorer] All current frontiers already visited. "
                  "Waiting for new ones …")
            time.sleep(IDLE_RETRY_SECS)
            continue

        tx, tz = target
        print(f"[Explorer] Navigating to frontier ({tx:.1f}, {tz:.1f})")

        # ── Navigate ─────────────────────────────────────────────────────────
        success = planner.navigate_to(agent_pos, tx, tz)

        if success:
            # Wait for the agent to physically arrive before marking visited.
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
            # Path planning failed — blacklist this frontier so we don't
            # spin on it forever.
            visited.append((tx, tz))
            print(f"[Explorer] Could not path to ({tx:.1f}, {tz:.1f}), skipping.")

        # Small pause before next cycle so the map can be updated with new
        # sensor frames collected while moving.
        time.sleep(0.5)


def _get_agent_pos():
    with _agent_pos_lock:
        return _agent_pos


def _wait_for_arrival(agent_pos, goal, timeout=60.0, poll=0.25):
    """
    Block until the agent is within FRONTIER_VISIT_RADIUS of goal.

    Abandons early if:
      - timeout expires, or
      - the distance to goal has been INCREASING for DIVERGE_WINDOW
        consecutive samples (agent is moving away / stuck on wrong side).

    Returns True if arrived, False if abandoned.
    """
    DIVERGE_WINDOW = 8   # consecutive rising samples before we bail

    gx, gz   = goal
    deadline = time.time() + timeout
    dist_history = []

    while time.time() < deadline:
        pos = agent_pos()
        if pos is not None:
            dist = np.hypot(pos[0] - gx, pos[2] - gz)

            if dist < FRONTIER_VISIT_RADIUS:
                return True

            dist_history.append(dist)

            # Only check divergence once we have enough samples.
            if len(dist_history) >= DIVERGE_WINDOW:
                window = dist_history[-DIVERGE_WINDOW:]
                # All consecutive differences positive → strictly increasing.
                if all(window[i] < window[i+1] for i in range(len(window)-1)):
                    print(f"[Explorer] Distance to ({gx:.1f}, {gz:.1f}) "
                          f"increasing for {DIVERGE_WINDOW} samples "
                          f"(last={dist:.2f}m) — abandoning goal.")
                    return False

        time.sleep(poll)

    print(f"[Explorer] Arrival timeout for goal ({gx:.1f}, {gz:.1f})")
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting perception server + autonomous exploration …")

    # Sensor server runs in its own daemon thread.
    t_sensor = threading.Thread(target=sensor_server, daemon=True)
    t_sensor.start()

    # Exploration loop runs in its own daemon thread so KeyboardInterrupt
    # on the main thread cleanly stops everything.
    t_explore = threading.Thread(target=exploration_loop, daemon=True)
    t_explore.start()

    # Keep the main thread alive.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")