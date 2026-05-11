# navigation/astar_planner.py
#
# Grid-based A* path planner that operates directly on the OccupancyMap grid.
# Returns a list of (world_x, world_z) waypoints from the agent's current
# position to a goal, then sends them one-by-one as move_to commands to Unity.
#
# Usage
# -----
#   planner = AStarPlanner(occupancy_map, unity_host='127.0.0.1', unity_port=5002)
#   planner.navigate_to(agent_pos, goal_wx, goal_wz)   # blocking per-waypoint
#
# Or call plan() to get just the waypoint list without sending.

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
        Shared occupancy map instance.
    unity_host : str
    unity_port : int
        TCP address of the Unity CommandReceiver.
    inflation_radius : float
        Metres to inflate obstacles before planning. Keeps the path away
        from walls. Should be >= robot radius.
    waypoint_spacing : int
        After finding the raw cell path, keep only every Nth cell as a
        waypoint (reduces the number of move_to commands).
    send_timeout : float
        Seconds to wait for each move_to to be acknowledged / completed
        before sending the next one.  Set to 0 for fire-and-forget.
    """

    def __init__(
        self,
        occ_map           : OccupancyMap,
        unity_host        : str   = '127.0.0.1',
        unity_port        : int   = 5002,
        inflation_radius  : float = 0.4,   # metres
        waypoint_spacing  : int   = 10,    # cells between waypoints
        send_timeout      : float = 0.0,   # seconds (0 = fire-and-forget)
    ):
        self.map              = occ_map
        self.unity_host       = unity_host
        self.unity_port       = unity_port
        self.inflation_radius = inflation_radius
        self.waypoint_spacing = waypoint_spacing
        self.send_timeout     = send_timeout

    # ── Obstacle inflation ────────────────────────────────────────────────────

    def _inflated_obstacle_mask(self):
        """
        Dilate the OCCUPIED layer by inflation_radius to create a clearance
        buffer. Planning treats inflated cells as impassable.
        """
        occ  = (self.map.grid == OCCUPIED).astype(np.uint8)
        r    = max(1, int(math.ceil(self.inflation_radius / self.map.resolution)))
        from navigation.occupancy_map import OccupancyMap
        import cv2
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        return cv2.dilate(occ, k, iterations=1).astype(bool)

    # ── A* core ───────────────────────────────────────────────────────────────

    def plan(self, agent_pos, goal_wx, goal_wz):
        """
        Compute an A* path from agent_pos to (goal_wx, goal_wz).

        Returns
        -------
        list of (world_x, world_z)  or  [] if no path found.
        """
        start_cell = self.map.world_to_cell(agent_pos[0], agent_pos[2])
        goal_cell  = self.map.world_to_cell(goal_wx, goal_wz)

        if start_cell is None or goal_cell is None:
            print("[A*] Start or goal out of map bounds.")
            return []

        blocked = self._inflated_obstacle_mask()

        # If goal is inside inflated obstacle space, pull it to the nearest
        # free, non-inflated cell before planning.
        gc, gr = goal_cell
        if blocked[gr, gc]:
            pulled = self._nearest_nonblocked(goal_cell, blocked)
            if pulled is None:
                print("[A*] Goal inside obstacle and no free cell nearby.")
                return []
            print(f"[A*] Goal pulled from {goal_cell} to {pulled}")
            goal_cell = pulled

        # Also nudge start if it somehow landed in blocked space.
        sc, sr = start_cell
        if blocked[sr, sc]:
            pulled = self._nearest_nonblocked(start_cell, blocked)
            if pulled is None:
                print("[A*] Start inside obstacle and no free cell nearby.")
                return []
            start_cell = pulled

        sc, sr = start_cell
        gc, gr = goal_cell

        # 8-connected neighbours
        neighbours = [(-1,-1),(0,-1),(1,-1),
                      (-1, 0),       (1, 0),
                      (-1, 1),(0, 1),(1, 1)]
        diag_cost  = math.sqrt(2)

        g_score  = {start_cell: 0.0}
        f_score  = {start_cell: self._heuristic(sc, sr, gc, gr)}
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
                # Treat UNKNOWN as passable but penalised (encourages known space)
                cell_cost = 1.0 if self.map.grid[nr, nc] == FREE else 3.0
                step_cost = (diag_cost if dc != 0 and dr != 0 else 1.0) * cell_cost

                neighbour  = (nc, nr)
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour]   = tentative_g
                    f_score[neighbour]   = tentative_g + self._heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_heap, (f_score[neighbour], neighbour))

        print("[A*] No path found.")
        return []

    def _heuristic(self, c0, r0, c1, r1):
        """Octile distance heuristic (admissible for 8-connected grid)."""
        dx = abs(c0 - c1)
        dy = abs(r0 - r1)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _reconstruct(self, came_from, current):
        """Trace back the path and convert to world coords, downsampled."""
        cells = []
        while current in came_from:
            cells.append(current)
            current = came_from[current]
        cells.append(current)
        cells.reverse()

        # Downsample: keep every Nth cell + always keep the last
        step  = max(1, self.waypoint_spacing)
        kept  = cells[::step]
        if cells[-1] not in kept:
            kept.append(cells[-1])

        return [self.map.cell_to_world(c, r) for c, r in kept]

    def _nearest_nonblocked(self, cell, blocked, max_search=30):
        """BFS to the nearest cell that is not in the inflated obstacle mask."""
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
        """BFS to find the closest FREE cell to `cell`."""
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
        Send a single move_to command to Unity CommandReceiver.
        x/y in the MoveCommand map to Unity Z/X respectively
        (matches the coord flip in command.cs: targetPosition = new Vector3(py, ..., px)).
        """
        cmd = json.dumps({"command": "move_to", "x": wz, "y": wx, "theta": theta})
        try:
            with socket.create_connection((self.unity_host, self.unity_port), timeout=2.0) as s:
                s.sendall((cmd + '\n').encode('utf-8'))
        except Exception as e:
            print(f"[A*] send_move_to failed: {e}")

    def navigate_to(self, agent_pos, goal_wx, goal_wz, theta=0.0):
        """
        Plan a path to (goal_wx, goal_wz) and send each waypoint to Unity.

        If send_timeout > 0, waits that many seconds between waypoints so
        the agent has time to reach each one before the next is issued.
        """
        waypoints = self.plan(agent_pos, goal_wx, goal_wz)
        if not waypoints:
            print(f"[A*] No path to ({goal_wx:.1f}, {goal_wz:.1f})")
            return False

        print(f"[A*] Navigating to ({goal_wx:.1f}, {goal_wz:.1f}) "
              f"via {len(waypoints)} waypoints")

        for i, (wx, wz) in enumerate(waypoints):
            # Heading: point towards next waypoint (or use supplied theta at end)
            if i + 1 < len(waypoints):
                nx, nz = waypoints[i + 1]
                heading = math.degrees(math.atan2(nx - wx, nz - wz))
            else:
                heading = theta
            self.send_move_to(wx, wz, heading)
            if self.send_timeout > 0:
                time.sleep(self.send_timeout)

        return True