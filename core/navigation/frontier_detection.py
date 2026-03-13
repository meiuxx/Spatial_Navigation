from collections import deque
import numpy as np
import navigation.utils as utils

class FrontierDetector:
    def __init__(self, grid):
        self.map_open = set()
        self.map_close = set()
        self.frontier_open = set()
        self.frontier_close = set()
        self.frontiers = []
        self.grid = grid
        self.rows, self.cols = grid.shape

    def _is_frontier(self, cell):
        r, c = cell
        if not self._is_free(cell):
            return False
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if self._is_unknown((nr, nc)):
                return True
        return False

    def _is_free(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def _is_unknown(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == -1

    def _has_free_neighbor(self, cell):
        r, c = cell
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if self._is_free((nr, nc)):
                return True
        return False

    def explorer(self, pose):
        queue_m = deque()
        queue_m.append(pose)
        self.map_open.add(pose)

        while queue_m:
            p = queue_m.popleft()

            if p in self.map_close:
                continue

            if self._is_frontier(p):
                queue_f = deque()
                new_frontier = set()
                queue_f.append(p)
                self.frontier_open.add(p)

                while queue_f:
                    q = queue_f.popleft()

                    if q in self.map_close or q in self.frontier_close:
                        continue

                    if self._is_frontier(q):
                        new_frontier.add(q)

                    rq, cq = q # q is a tuple of (row, col), and it's a candidate frontier
                    #loop over neighborhood
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        # compute actual neighbor coordinates
                        nr, nc = rq + dr, cq + dc
                        w = (nr, nc)
                        
                        # no unknown or occupied neighbors allowed in BFS expansion
                        if not self._is_free(w):
                            continue

                        if (w not in self.frontier_open and
                            w not in self.frontier_close and
                            w not in self.map_close):
                            queue_f.append(w)
                            self.frontier_open.add(w)

                    self.frontier_close.add(q)

                self.frontiers.append(new_frontier)

                for cell in new_frontier:
                    self.map_close.add(cell)

            rp, cp = p
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = rp + dr, cp + dc
                v = (nr, nc)

                if not self._is_free(v):
                    continue

                if v not in self.map_open and v not in self.map_close:
                    if self._has_free_neighbor(v):
                        queue_m.append(v)
                        self.map_open.add(v)

            self.map_close.add(p)

        return self.frontiers
    
    # Helper function to select the nearest frontier centroid
def select_frontier(frontiers, robot_world_x, robot_world_y, map_size_pixels, map_size_meters):
    if not frontiers:
        return None
    best_dist = float('inf')
    best_centroid = None
    for region in frontiers:
        if not region:
            continue
        rows, cols = zip(*region)
        centroid_r = int(np.mean(rows))
        centroid_c = int(np.mean(cols))
        world_x, world_y = utils.pixels_to_world(
            centroid_c, centroid_r, map_size_pixels, map_size_meters
        )
        dist = np.hypot(world_x - robot_world_x, world_y - robot_world_y)
        if dist < best_dist:
            best_dist = dist
            best_centroid = (world_x, world_y)
    return best_centroid