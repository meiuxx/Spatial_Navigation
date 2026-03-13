import heapq
import numpy as np
from . import utils

class AStar:
    def __init__(self, grid):
        """
        grid: 2D numpy array with values: 1=free, 0=occupied, -1=unknown.
        Unknown cells are treated as free for planning (since we can move through them,
        but they might later become occupied). If you prefer to avoid unknown,
        change the condition below.
        """
        self.grid = grid
        self.rows, self.cols = grid.shape

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])   # Manhattan distance

    def neighbors(self, node):
        r, c = node
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Allow moving through free (1) and unknown (-1) cells
                if self.grid[nr, nc] != 0:   # not occupied
                    yield (nr, nc)

    def search(self, start, goal):
        """
        Returns a list of (r,c) tuples from start to goal (inclusive),
        or None if no path exists.
        """
        if self.grid[start[0], start[1]] == 0 or self.grid[goal[0], goal[1]] == 0:
            return None   # start or goal is occupied

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None