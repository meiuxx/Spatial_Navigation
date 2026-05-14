# navigation/astar_planner.py
#
# Grid-based A* path planner that operates directly on the OccupancyMap grid.
# Returns a list of (world_x, world_z) waypoints from the agent's current
# position to a goal, then sends them one-by-one as move_to commands to Unity.

import heapq
import socket
import json
import time
import math
import numpy as np
from navigation.occupancy_map import OccupancyMap, FREE, OCCUPIED, UNKNOWN


class AStarPlanner:
    """
    A* path planner on an OccupancyMap grid.

    Parameters
    ----------
    occ_map : OccupancyMap
    unity_host : str
    unity_port : int
    inflation_radius : float
    waypoint_spacing : int
    send_timeout : float
        Seconds to wait between waypoints (0 = fire-and-forget).
    """

    def __init__(
        self,
        occ_map           : OccupancyMap,
        unity_host        : str   = '127.0.0.1',
        unity_port        : int   = 5002,
        inflation_radius  : float = 0.4,
        waypoint_spacing  : int   = 10,
        send_timeout      : float = 0.0,
    ):
        self.map              = occ_map
        self.unity_host       = unity_host
        self.unity_port       = unity_port
        self.inflation_radius = inflation_radius
        self.waypoint_spacing = waypoint_spacing
        self.send_timeout     = send_timeout

        # FIX (Bug 2): Persistent socket reused across all waypoints in a
        # navigate_to call.  CommandReceiver.cs accepts one client at a time
        # in a blocking loop — opening a fresh connection per waypoint races
        # against the server's AcceptTcpClient cycle and causes dropped commands.
        self._sock   = None
        self._sock_lock = __import__('threading').Lock()

    # ── Persistent connection helpers ─────────────────────────────────────────

    def _ensure_connected(self):
        """Open the persistent socket if it isn't already open."""
        if self._sock is not None:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.unity_host, self.unity_port))
            s.settimeout(5.0)
            self._sock = s
            print(f"[A*] Connected to Unity on {self.unity_host}:{self.unity_port}")
            return True
        except Exception as e:
            print(f"[A*] Could not connect to Unity: {e}")
            self._sock = None
            return False

    def _close_connection(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # ── Obstacle inflation ────────────────────────────────────────────────────

    def _inflated_obstacle_mask(self):
        occ  = (self.map.grid == OCCUPIED).astype(np.uint8)
        r    = max(1, int(math.ceil(self.inflation_radius / self.map.resolution)))
        import cv2
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        return cv2.dilate(occ, k, iterations=1).astype(bool)

    # ── A* core ───────────────────────────────────────────────────────────────

    def plan(self, agent_pos, goal_wx, goal_wz):
        """
        Compute an A* path from agent_pos to (goal_wx, goal_wz).
        Returns list of (world_x, world_z) or [] if no path found.
        """
        start_cell = self.map.world_to_cell(agent_pos[0], agent_pos[2])
        goal_cell  = self.map.world_to_cell(goal_wx, goal_wz)

        if start_cell is None or goal_cell is None:
            print("[A*] Start or goal out of map bounds.")
            return []

        blocked = self._inflated_obstacle_mask()

        gc, gr = goal_cell
        if blocked[gr, gc]:
            pulled = self._nearest_nonblocked(goal_cell, blocked)
            if pulled is None:
                print("[A*] Goal inside obstacle and no free cell nearby.")
                return []
            print(f"[A*] Goal pulled from {goal_cell} to {pulled}")
            goal_cell = pulled

        sc, sr = start_cell
        if blocked[sr, sc]:
            pulled = self._nearest_nonblocked(start_cell, blocked)
            if pulled is None:
                print("[A*] Start inside obstacle and no free cell nearby.")
                return []
            start_cell = pulled

        sc, sr = start_cell
        gc, gr = goal_cell

        neighbours = [(-1,-1),(0,-1),(1,-1),
                      (-1, 0),       (1, 0),
                      (-1, 1),(0, 1),(1, 1)]
        diag_cost  = math.sqrt(2)

        g_score   = {start_cell: 0.0}
        f_score   = {start_cell: self._heuristic(sc, sr, gc, gr)}
        came_from = {}
        open_heap = [(f_score[start_cell], start_cell)]

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_cell:
                return self._reconstruct(came_from, current)

            cc, cr = current
            for dc, dr in neighbours:
                nc, nr = cc + dc, cr + dr
                if not (0 <= nc < self.map.grid_w and 0 <= nr < self.map.grid_h):
                    continue
                if blocked[nr, nc]:
                    continue
                cell_cost  = 1.0 if self.map.grid[nr, nc] == FREE else 3.0
                step_cost  = (diag_cost if dc != 0 and dr != 0 else 1.0) * cell_cost

                neighbour    = (nc, nr)
                tentative_g  = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour]   = tentative_g
                    f_score[neighbour]   = tentative_g + self._heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_heap, (f_score[neighbour], neighbour))

        print("[A*] No path found.")
        return []

    def _heuristic(self, c0, r0, c1, r1):
        dx = abs(c0 - c1)
        dy = abs(r0 - r1)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _reconstruct(self, came_from, current):
        cells = []
        while current in came_from:
            cells.append(current)
            current = came_from[current]
        cells.append(current)
        cells.reverse()

        step = max(1, self.waypoint_spacing)
        kept = cells[::step]
        if cells[-1] not in kept:
            kept.append(cells[-1])

        return [self.map.cell_to_world(c, r) for c, r in kept]

    def _nearest_nonblocked(self, cell, blocked, max_search=30):
        from collections import deque
        visited = {cell}
        q = deque([cell])
        while q:
            c, r = q.popleft()
            if not blocked[r, c]:
                return (c, r)
            for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                nc, nr = c + dc, r + dr
                nb = (nc, nr)
                if (0 <= nc < self.map.grid_w and 0 <= nr < self.map.grid_h
                        and nb not in visited and len(visited) < max_search**2):
                    visited.add(nb)
                    q.append(nb)
        return None

    def _nearest_free(self, cell, max_search=20):
        from collections import deque
        visited = {cell}
        q = deque([cell])
        while q:
            c, r = q.popleft()
            if self.map.grid[r, c] == FREE:
                return (c, r)
            for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
                nc, nr = c+dc, r+dr
                nb = (nc, nr)
                if (0 <= nc < self.map.grid_w and 0 <= nr < self.map.grid_h
                        and nb not in visited):
                    visited.add(nb)
                    if len(visited) < max_search * max_search:
                        q.append(nb)
        return None

    # ── Unity command sender ──────────────────────────────────────────────────

    def send_move_to(self, wx, wz, theta=0.0):
        """
        Send a single move_to command over the persistent socket.

        FIX (Bug 2): Instead of opening a new TCP connection per waypoint
        (which races with CommandReceiver.cs's single-threaded AcceptTcpClient
        loop and drops commands), we reuse one connection for the full path.
        If the socket is broken we reconnect once and retry.
        """
        cmd = json.dumps({"command": "move_to", "x": wz, "y": wx, "theta": theta}) + '\n'
        with self._sock_lock:
            for attempt in range(2):
                if not self._ensure_connected():
                    return
                try:
                    self._sock.sendall(cmd.encode('utf-8'))
                    return   # success
                except Exception as e:
                    print(f"[A*] send_move_to failed (attempt {attempt+1}): {e}")
                    self._close_connection()
                    # retry once after reconnect

    def navigate_to(self, agent_pos, goal_wx, goal_wz, theta=0.0):
        """
        Plan a path to (goal_wx, goal_wz) and send each waypoint to Unity
        over a single persistent connection.
        """
        waypoints = self.plan(agent_pos, goal_wx, goal_wz)
        if not waypoints:
            print(f"[A*] No path to ({goal_wx:.1f}, {goal_wz:.1f})")
            return False

        print(f"[A*] Navigating to ({goal_wx:.1f}, {goal_wz:.1f}) "
              f"via {len(waypoints)} waypoints")

        for i, (wx, wz) in enumerate(waypoints):
            if i + 1 < len(waypoints):
                nx, nz  = waypoints[i + 1]
                heading = math.degrees(math.atan2(nx - wx, nz - wz))
            else:
                heading = theta
            self.send_move_to(wx, wz, heading)
            if self.send_timeout > 0:
                time.sleep(self.send_timeout)

        return True