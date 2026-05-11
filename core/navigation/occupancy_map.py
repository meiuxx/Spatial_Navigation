# navigation/occupancy_map.py
#
# Builds a 2-D top-down occupancy grid from per-frame depth maps.
# Coordinate conventions match the rest of the project:
#   - Unity is LEFT-HANDED  (Y up, Z forward)
#   - Internally we convert to RIGHT-HANDED for maths, then back.
#   - The 2-D grid lives in the Unity X-Z plane (horizontal floor plane).
#
# Cell states
#   UNKNOWN   = 0   (grey)
#   FREE      = 1   (white)
#   OCCUPIED  = 2   (black)
#
# Frontiers = clusters of FREE cells that border at least one UNKNOWN cell.

import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ── Cell state constants ──────────────────────────────────────────────────────
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


class OccupancyMap:
    """
    2-D occupancy grid built from depth images sent by Unity.

    Parameters
    ----------
    resolution : float
        Metres per cell (e.g. 0.1 → 10 cm/cell).
    width_m, height_m : float
        Physical extent of the map in metres.  The origin (0, 0) in world space
        maps to the centre of the grid.
    max_depth : float
        Depth readings beyond this value are discarded (likely sky / far wall).
    col_step, row_step : int
        Sub-sample the depth image every N columns / rows for speed.
    floor_band : float
        Only depth points whose world-Y is within ±floor_band of the camera
        height are used.  This strips out ceiling and floor hits.
    obstacle_height_min : float
        World-Y lower bound (relative to cam_pos_y) for a point to count as an
        obstacle.  Points below this are floor; points above are ceiling.
    obstacle_height_max : float
        World-Y upper bound (relative to cam_pos_y) for obstacle classification.
    free_ray_samples : int
        How many evenly-spaced cells to mark FREE along each depth ray
        (between camera and the hit point).
    hit_threshold : int
        How many times a cell must be hit before it is marked OCCUPIED.
        Set to 1 for aggressive mapping, higher for noise robustness.
    morph_free_radius : int
        Morphological closing radius (cells) applied to the FREE layer each
        render to fill small holes and connect nearby free regions.
        Set to 0 to disable.
    morph_occ_radius : int
        Morphological closing radius (cells) applied to the OCCUPIED layer.
        Bridges small gaps in walls so frontiers don't leak through them.
        Set to 0 to disable.
    frontier_min_size : int
        Frontier clusters with fewer than this many cells are discarded as
        noise.
    frontier_merge_radius : float
        After clustering, frontier centroids closer than this distance (metres)
        are merged into a single point.  Reduces redundant navigation goals.
    """

    def __init__(
        self,
        resolution            = 0.1,
        width_m               = 60.0,
        height_m              = 60.0,
        max_depth             = 15.0,
        col_step              = 4,
        row_step              = 4,
        floor_band            = 1.2,
        obstacle_height_min   = 0.3,
        obstacle_height_max   = 2.2,
        free_ray_samples      = 8,
        hit_threshold         = 1,
        # ── new smoothing / clustering params ────────────────────────────────
        morph_free_radius     = 3,   # cells  (~30 cm at 10 cm/cell)
        morph_occ_radius      = 2,   # cells  (~20 cm)
        frontier_min_size     = 5,   # cells
        frontier_merge_radius = 1.5, # metres
    ):
        self.resolution            = resolution
        self.width_m               = width_m
        self.height_m              = height_m
        self.max_depth             = max_depth
        self.col_step              = col_step
        self.row_step              = row_step
        self.floor_band            = floor_band
        self.obstacle_height_min   = obstacle_height_min
        self.obstacle_height_max   = obstacle_height_max
        self.free_ray_samples      = free_ray_samples
        self.hit_threshold         = hit_threshold
        self.morph_free_radius     = morph_free_radius
        self.morph_occ_radius      = morph_occ_radius
        self.frontier_min_size     = frontier_min_size
        self.frontier_merge_radius = frontier_merge_radius

        # Grid dimensions in cells
        self.grid_w = int(np.ceil(width_m  / resolution))
        self.grid_h = int(np.ceil(height_m / resolution))

        self.origin_x = -width_m  / 2.0
        self.origin_z = -height_m / 2.0

        # State grid + hit counter
        self.grid       = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        self._hit_count = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

        # Pre-allocate visualisation image (BGR)
        self._vis = np.full((self.grid_h, self.grid_w, 3), 128, dtype=np.uint8)

        print(f"[OccupancyMap] grid {self.grid_w}×{self.grid_h} cells "
              f"({width_m:.0f}m × {height_m:.0f}m @ {resolution*100:.0f} cm/cell)")

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def world_to_cell(self, wx, wz):
        col = int((wx - self.origin_x) / self.resolution)
        row = int((wz - self.origin_z) / self.resolution)
        if 0 <= col < self.grid_w and 0 <= row < self.grid_h:
            return col, row
        return None

    def cell_to_world(self, col, row):
        wx = self.origin_x + (col + 0.5) * self.resolution
        wz = self.origin_z + (row + 0.5) * self.resolution
        return wx, wz

    # ── Morphological helpers ─────────────────────────────────────────────────

    def _make_disk(self, radius):
        """Return a circular morphological kernel with the given cell radius."""
        d = 2 * radius + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        return k

    def _smoothed_layers(self):
        """
        Return morphologically-closed FREE and OCCUPIED masks.

        Closing = dilation then erosion: fills small holes / gaps inside a
        region without expanding its outer boundary much.  We never let
        smoothed-OCCUPIED overwrite cells the raw grid marks as FREE, and
        vice versa, so the operation is conservative.
        """
        free_raw = (self.grid == FREE).astype(np.uint8)
        occ_raw  = (self.grid == OCCUPIED).astype(np.uint8)

        if self.morph_free_radius > 0:
            k        = self._make_disk(self.morph_free_radius)
            free_smooth = cv2.morphologyEx(free_raw, cv2.MORPH_CLOSE, k)
            # Don't let smoothed FREE bleed into raw OCCUPIED cells
            free_smooth[occ_raw == 1] = 0
        else:
            free_smooth = free_raw

        if self.morph_occ_radius > 0:
            k       = self._make_disk(self.morph_occ_radius)
            occ_smooth = cv2.morphologyEx(occ_raw, cv2.MORPH_CLOSE, k)
            # Don't let smoothed OCCUPIED overwrite raw FREE cells
            occ_smooth[free_raw == 1] = 0
        else:
            occ_smooth = occ_raw

        return free_smooth, occ_smooth

    # ── Main update ───────────────────────────────────────────────────────────

    def clear_near_agent(self, agent_pos, radius_m=1.2):
        """
        Explicitly mark cells within `radius_m` of the agent as FREE
        (unless they are OCCUPIED by an obstacle).

        This prevents stale frontier cells from persisting after the agent
        has physically passed through them — morphological smoothing can
        otherwise keep UNKNOWN islands alive near the agent path.
        """
        cell = self.world_to_cell(agent_pos[0], agent_pos[2])
        if cell is None:
            return
        r_cells = max(1, int(math.ceil(radius_m / self.resolution)))
        cc, cr  = cell
        c0 = max(0, cc - r_cells)
        c1 = min(self.grid_w - 1, cc + r_cells)
        r0 = max(0, cr - r_cells)
        r1 = min(self.grid_h - 1, cr + r_cells)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (c - cc)**2 + (r - cr)**2 <= r_cells**2:
                    if self.grid[r, c] != OCCUPIED:
                        self.grid[r, c] = FREE

    def update(self, depth_linear, depth_w, depth_h,
               rgb_w, rgb_h, fov, cam_pos, cam_rot):
        """
        Integrate one depth frame into the occupancy grid.
        """
        fov_rad = np.radians(fov)
        fy = depth_h / (2.0 * np.tan(fov_rad / 2.0))
        fx = fy
        cx_img = depth_w / 2.0
        cy_img = depth_h / 2.0

        cam_pos_rh = np.array([ cam_pos[0],  cam_pos[1], -cam_pos[2]])
        cam_rot_rh = np.array([ cam_rot[0], -cam_rot[1],  cam_rot[2], cam_rot[3]])
        rot = R.from_quat(cam_rot_rh)

        cam_cell = self.world_to_cell(cam_pos[0], cam_pos[2])

        rows = np.arange(0, depth_h, self.row_step)
        cols = np.arange(0, depth_w, self.col_step)
        col_grid, row_grid = np.meshgrid(cols, rows)

        depths = depth_linear[row_grid, col_grid]
        valid  = (depths > 0.05) & (depths < self.max_depth)

        col_v  = col_grid[valid].astype(float)
        row_v  = row_grid[valid].astype(float)
        dist_v = depths[valid]

        if dist_v.size == 0:
            return

        Z =  dist_v
        X = (col_v - cx_img) * Z / fx
        Y = (row_v - cy_img) * Z / fy

        pts_cam_rh   = np.stack([ X, -Y, -Z], axis=1)
        pts_world_rh = rot.apply(pts_cam_rh) + cam_pos_rh

        wx = pts_world_rh[:, 0]
        wy = pts_world_rh[:, 1]
        wz = -pts_world_rh[:, 2]

        rel_y      = wy - cam_pos[1]
        height_ok  = (rel_y > -self.floor_band) & (rel_y < self.floor_band)
        is_obstacle = (rel_y > self.obstacle_height_min) & (rel_y < self.obstacle_height_max)

        wx          = wx[height_ok]
        wz          = wz[height_ok]
        is_obstacle = is_obstacle[height_ok]

        for i in range(len(wx)):
            hit_cell = self.world_to_cell(wx[i], wz[i])
            if hit_cell is None:
                continue

            if cam_cell is not None:
                self._mark_ray_free(cam_cell, hit_cell)

            if is_obstacle[i]:
                self._hit_count[hit_cell[1], hit_cell[0]] += 1
                if self._hit_count[hit_cell[1], hit_cell[0]] >= self.hit_threshold:
                    self.grid[hit_cell[1], hit_cell[0]] = OCCUPIED
            else:
                if self.grid[hit_cell[1], hit_cell[0]] != OCCUPIED:
                    self.grid[hit_cell[1], hit_cell[0]] = FREE

    def _mark_ray_free(self, cam_cell, hit_cell):
        n    = self.free_ray_samples
        cols = np.linspace(cam_cell[0], hit_cell[0], n + 2)[1:-1].astype(int)
        rows = np.linspace(cam_cell[1], hit_cell[1], n + 2)[1:-1].astype(int)
        cols = np.clip(cols, 0, self.grid_w - 1)
        rows = np.clip(rows, 0, self.grid_h - 1)
        for c, r in zip(cols, rows):
            if self.grid[r, c] != OCCUPIED:
                self.grid[r, c] = FREE

    # ── Frontier detection ────────────────────────────────────────────────────

    def _raw_frontier_mask(self, free_mask=None, unknown_mask=None):
        """
        Compute the per-cell frontier mask from (optionally smoothed) layers.
        """
        if free_mask is None:
            free_mask    = (self.grid == FREE).astype(np.uint8)
        if unknown_mask is None:
            unknown_mask = (self.grid == UNKNOWN).astype(np.uint8)

        kernel   = np.ones((3, 3), dtype=np.uint8)
        dilated  = cv2.dilate(unknown_mask, kernel, iterations=1)
        return (free_mask & dilated).astype(np.uint8)

    def _safe_clearance_mask(self, extra_cells=2):
        """
        Returns a boolean mask that is True for cells that are:
          - FREE (raw grid), AND
          - not within (inflation_radius + extra_cells) of any OCCUPIED cell.

        Used to validate / pull frontier centroids away from walls.
        `extra_cells` adds a small extra buffer on top of the planner's
        inflation radius so centroids have guaranteed manoeuvring room.
        """
        occ  = (self.grid == OCCUPIED).astype(np.uint8)
        # Use morph_occ_radius as a proxy for the planner's inflation radius,
        # plus the requested extra buffer.
        r    = self.morph_occ_radius + extra_cells
        k    = self._make_disk(r)
        inflated = cv2.dilate(occ, k, iterations=1).astype(bool)
        free_raw = self.grid == FREE
        return free_raw & ~inflated   # True = safely traversable

    def _pull_to_safe(self, col, row, safe_mask, max_search=40):
        """
        If (col, row) is not in safe_mask, BFS outward until a safe cell
        is found.  Returns (col, row) unchanged if it is already safe,
        or the nearest safe cell, or None if nothing found within max_search.
        """
        if safe_mask[row, col]:
            return col, row
        from collections import deque
        visited = set()
        q = deque([(col, row, 0)])
        while q:
            c, r, depth = q.popleft()
            if (c, r) in visited:
                continue
            visited.add((c, r))
            if safe_mask[r, c]:
                return c, r
            if depth >= max_search:
                continue
            for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                nc, nr = c + dc, r + dr
                if (0 <= nc < self.grid_w and 0 <= nr < self.grid_h
                        and (nc, nr) not in visited):
                    q.append((nc, nr, depth + 1))
        return None

    def get_frontiers(self):
        """
        Return a list of (world_x, world_z) cluster centroids, guaranteed
        to lie in safely traversable space (away from walls).

        Pipeline
        --------
        1. Morphologically smooth FREE / OCCUPIED layers.
        2. Build a safe-clearance mask (FREE and not near any wall).
        3. Compute per-cell frontier mask on the smoothed layers, then
           REMOVE any frontier cell that falls outside the safe mask —
           this strips frontier pixels that are inside walls or right
           against them before clustering.
        4. Find connected components; discard small ones.
        5. Compute centroid of each cluster; if centroid lands in unsafe
           space, pull it to the nearest safe cell via BFS.
        6. Discard any centroid that could not be pulled to a safe cell.
        7. Merge remaining centroids within frontier_merge_radius.
        """
        free_smooth, occ_smooth = self._smoothed_layers()
        unknown_smooth = ((free_smooth == 0) & (occ_smooth == 0)).astype(np.uint8)

        frontier_mask = self._raw_frontier_mask(free_smooth, unknown_smooth)

        # ── Strip frontier cells that are inside / too close to walls ────────
        safe_mask = self._safe_clearance_mask(extra_cells=2)
        # A frontier cell is only kept if it (or its immediate 1-cell
        # neighbourhood) has at least one safe cell — i.e. it is reachable
        # from safe space.  A strict AND would discard valid thin corridors,
        # so we dilate the safe mask by 1 before masking.
        safe_dilated = cv2.dilate(safe_mask.astype(np.uint8),
                                  np.ones((3, 3), np.uint8), iterations=1)
        frontier_mask = (frontier_mask.astype(bool) & safe_dilated.astype(bool)).astype(np.uint8)

        # ── Connected-component clustering ───────────────────────────────────
        n_labels, label_img = cv2.connectedComponents(frontier_mask, connectivity=8)

        centroids = []
        for lbl in range(1, n_labels):
            component = (label_img == lbl)
            size = int(component.sum())
            if size < self.frontier_min_size:
                continue

            rows, cols = np.where(component)

            # Use the component cell closest to its geometric mean as the
            # centroid seed — more stable than the raw float mean.
            mean_col = float(cols.mean())
            mean_row = float(rows.mean())
            dists    = (cols - mean_col)**2 + (rows - mean_row)**2
            best_idx = int(np.argmin(dists))
            seed_col = int(cols[best_idx])
            seed_row = int(rows[best_idx])

            # Pull the seed to safe space if needed.
            result = self._pull_to_safe(seed_col, seed_row, safe_mask)
            if result is None:
                continue   # no safe cell reachable — discard cluster

            safe_col, safe_row = result
            wx, wz = self.cell_to_world(safe_col, safe_row)
            centroids.append((wx, wz, size))

        if not centroids:
            return []

        merged = self._merge_centroids(centroids)
        return merged

    def _merge_centroids(self, centroids):
        """
        Greedily merge centroids that are within ``frontier_merge_radius``
        metres of each other.  Larger clusters absorb smaller ones.
        Returns list of (world_x, world_z).
        """
        r = self.frontier_merge_radius
        # Sort largest-first so big clusters act as seeds
        remaining = sorted(centroids, key=lambda c: -c[2])
        merged    = []

        while remaining:
            seed     = remaining.pop(0)
            sx, sz   = seed[0], seed[1]
            total_w  = seed[2]
            sum_x    = sx * total_w
            sum_z    = sz * total_w

            still_remaining = []
            for c in remaining:
                dist = np.hypot(c[0] - sx, c[1] - sz)
                if dist <= r:
                    sum_x   += c[0] * c[2]
                    sum_z   += c[1] * c[2]
                    total_w += c[2]
                else:
                    still_remaining.append(c)

            remaining = still_remaining
            merged.append((sum_x / total_w, sum_z / total_w))

        return merged

    def get_frontier_count(self):
        return len(self.get_frontiers())

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _vis_init(self, window_name):
        if hasattr(self, '_vis_win_name'):
            return

        self._vis_win_name   = window_name
        self._vis_zoom       = max(1.0, 600.0 / self.grid_w)
        self._vis_pan        = [self.grid_w / 2.0, self.grid_h / 2.0]
        self._vis_dragging   = False
        self._vis_drag_start = (0, 0)
        self._vis_pan_start  = [0.0, 0.0]
        self._vis_show_heat  = False
        self._vis_trail      = []

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 700, 720)

        def _mouse(event, mx, my, flags, param):
            z = self._vis_zoom
            if event == cv2.EVENT_MOUSEWHEEL:
                win_w = int(self.grid_w * z)
                win_h = int(self.grid_h * z)
                cx_before = self._vis_pan[0] + (mx - win_w / 2) / z
                cy_before = self._vis_pan[1] + (my - win_h / 2) / z
                factor = 1.15 if flags > 0 else (1.0 / 1.15)
                self._vis_zoom = float(np.clip(z * factor, 0.5, 20.0))
                z2 = self._vis_zoom
                self._vis_pan[0] = cx_before - (mx - win_w / 2) / z2
                self._vis_pan[1] = cy_before - (my - win_h / 2) / z2
            elif event == cv2.EVENT_LBUTTONDOWN:
                self._vis_dragging   = True
                self._vis_drag_start = (mx, my)
                self._vis_pan_start  = list(self._vis_pan)
            elif event == cv2.EVENT_MOUSEMOVE and self._vis_dragging:
                dx = (mx - self._vis_drag_start[0]) / self._vis_zoom
                dy = (my - self._vis_drag_start[1]) / self._vis_zoom
                self._vis_pan[0] = self._vis_pan_start[0] - dx
                self._vis_pan[1] = self._vis_pan_start[1] - dy
            elif event == cv2.EVENT_LBUTTONUP:
                self._vis_dragging = False

        cv2.setMouseCallback(window_name, _mouse)

    def _build_base_layer(self):
        """
        Render the full grid at 1 px/cell.

        Uses smoothed FREE/OCCUPIED layers for display so the rendered map
        matches the frontier computation (filled holes, bridged wall gaps).
        Raw frontier cells are shown as cyan; cluster centroids as large
        magenta circles.
        """
        base = np.full((self.grid_h, self.grid_w, 3), 60, dtype=np.uint8)

        free_smooth, occ_smooth = self._smoothed_layers()
        base[free_smooth == 1]  = (235, 235, 235)
        base[occ_smooth  == 1]  = ( 30,  30,  30)

        # Heatmap overlay (raw OCCUPIED cells only)
        if self._vis_show_heat:
            occ_raw = self.grid == OCCUPIED
            if occ_raw.any():
                hc      = self._hit_count.astype(float)
                hc_max  = hc[occ_raw].max()
                if hc_max > 0:
                    norm        = np.clip(hc / hc_max, 0, 1)
                    heat_uint8  = (norm * 255).astype(np.uint8)
                    heat_color  = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
                    base[occ_raw] = heat_color[occ_raw]

        # Per-cell frontier pixels (cyan) — computed on smoothed layers
        unknown_smooth = ((free_smooth == 0) & (occ_smooth == 0)).astype(np.uint8)
        frontier_mask  = self._raw_frontier_mask(free_smooth, unknown_smooth)
        base[frontier_mask.astype(bool)] = (0, 220, 220)

        # Cluster centroids — filled magenta circles
        centroids = self.get_frontiers()
        for wx, wz in centroids:
            cell = self.world_to_cell(wx, wz)
            if cell is not None:
                cv2.circle(base, (cell[0], cell[1]),
                           max(3, int(self.frontier_merge_radius / self.resolution * 0.4)),
                           (220, 0, 220), -1, cv2.LINE_AA)

        return base, frontier_mask, centroids

    def _apply_zoom_pan(self, base):
        z  = self._vis_zoom
        cx = self._vis_pan[0]
        cy = self._vis_pan[1]
        win_w_cells = 700 / z
        win_h_cells = 700 / z
        col0 = int(np.clip(cx - win_w_cells / 2, 0, self.grid_w))
        col1 = int(np.clip(cx + win_w_cells / 2, 0, self.grid_w))
        row0 = int(np.clip(cy - win_h_cells / 2, 0, self.grid_h))
        row1 = int(np.clip(cy + win_h_cells / 2, 0, self.grid_h))
        crop = base[row0:row1, col0:col1]
        if crop.size == 0:
            return base, (0, 0)
        out_w = max(int((col1 - col0) * z), 1)
        out_h = max(int((row1 - row0) * z), 1)
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST), (col0, row0)

    def _draw_trail(self, canvas, col0, row0):
        z   = self._vis_zoom
        if len(self._vis_trail) < 2:
            return
        pts = self._vis_trail[-300:]
        n   = len(pts)
        for i in range(1, n):
            alpha = i / n
            color = (int(50 + 205 * alpha), int(100 * (1 - alpha)), int(200 * alpha))
            c0, r0 = pts[i-1]
            c1, r1 = pts[i]
            px0 = int((c0 - col0) * z + z / 2)
            py0 = int((r0 - row0) * z + z / 2)
            px1 = int((c1 - col0) * z + z / 2)
            py1 = int((r1 - row0) * z + z / 2)
            cv2.line(canvas, (px0, py0), (px1, py1), color, 1, cv2.LINE_AA)

    def _draw_agent(self, canvas, agent_cell, col0, row0):
        z   = self._vis_zoom
        c, r = agent_cell
        px  = int((c - col0) * z + z / 2)
        py  = int((r - row0) * z + z / 2)
        radius = max(4, int(z * 1.5))
        cv2.circle(canvas, (px, py), radius,     (0, 0, 220),   -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_grid_lines(self, canvas, col0, row0):
        z = self._vis_zoom
        cells_per_10m = int(10.0 / self.resolution)
        if z * cells_per_10m < 20:
            return
        h, w = canvas.shape[:2]
        start_col = (col0 // cells_per_10m) * cells_per_10m
        c = start_col
        while c < col0 + w / z:
            px = int((c - col0) * z)
            cv2.line(canvas, (px, 0), (px, h), (100, 100, 100), 1)
            wx = self.origin_x + c * self.resolution
            cv2.putText(canvas, f"{wx:.0f}m", (px + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
            c += cells_per_10m
        start_row = (row0 // cells_per_10m) * cells_per_10m
        r = start_row
        while r < row0 + h / z:
            py = int((r - row0) * z)
            cv2.line(canvas, (0, py), (w, py), (100, 100, 100), 1)
            wz = self.origin_z + r * self.resolution
            cv2.putText(canvas, f"{wz:.0f}m", (2, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
            r += cells_per_10m

    def _draw_hud(self, canvas, frontier_count, agent_pos):
        h, w  = canvas.shape[:2]
        panel_h = 80
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        total_known  = int(np.sum(self.grid != UNKNOWN))
        total_cells  = self.grid_w * self.grid_h
        explored_pct = 100.0 * total_known / total_cells if total_cells > 0 else 0.0

        line1 = (f"Frontier clusters: {frontier_count}   "
                 f"Explored: {explored_pct:.1f}%   "
                 f"Zoom: {self._vis_zoom:.1f}x   "
                 f"{'[HEAT]' if self._vis_show_heat else ''}")
        if agent_pos is not None:
            line1 += f"   Agent: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}, {agent_pos[2]:.1f})"

        line2 = "Scroll=zoom   Drag=pan   H=heatmap   R=reset view   S=save PNG"

        cv2.putText(canvas, line1, (8, h - panel_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (8, h - panel_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

        items = [
            ((235, 235, 235), "Free"),
            (( 30,  30,  30), "Occupied"),
            (( 60,  60,  60), "Unknown"),
            ((  0, 220, 220), "Frontier"),
            ((220,   0, 220), "F-Cluster"),
            ((  0,   0, 220), "Agent"),
        ]
        x = 8
        y = h - panel_h + 62
        for color, label in items:
            cv2.rectangle(canvas, (x, y - 9), (x + 12, y + 3), color, -1)
            cv2.rectangle(canvas, (x, y - 9), (x + 12, y + 3), (180, 180, 180), 1)
            cv2.putText(canvas, label, (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            x += 75

    def render(self, agent_pos=None, window_name="Occupancy Map", scale=None):
        """
        Rich interactive visualiser.

        Controls:
            Scroll  – zoom in / out
            Drag    – pan
            H       – toggle hit-count heatmap
            R       – reset view
            S       – save PNG
        """
        self._vis_init(window_name)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('h') or key == ord('H'):
            self._vis_show_heat = not self._vis_show_heat
            print(f"[OccupancyMap] Heatmap {'ON' if self._vis_show_heat else 'OFF'}")
        elif key == ord('r') or key == ord('R'):
            self._vis_zoom = max(1.0, 600.0 / self.grid_w)
            self._vis_pan  = [self.grid_w / 2.0, self.grid_h / 2.0]
            print("[OccupancyMap] View reset")
        elif key == ord('s') or key == ord('S'):
            import time as _time
            self.save(f"occupancy_{int(_time.time())}.png")

        if agent_pos is not None:
            ac = self.world_to_cell(agent_pos[0], agent_pos[2])
            if ac is not None and (not self._vis_trail or self._vis_trail[-1] != ac):
                self._vis_trail.append(ac)

        base, frontier_mask, centroids = self._build_base_layer()
        frontier_count = len(centroids)

        canvas, (col0, row0) = self._apply_zoom_pan(base)
        canvas = canvas.copy()

        self._draw_grid_lines(canvas, col0, row0)
        self._draw_trail(canvas, col0, row0)

        if agent_pos is not None:
            ac = self.world_to_cell(agent_pos[0], agent_pos[2])
            if ac is not None:
                self._draw_agent(canvas, ac, col0, row0)

        hud_canvas = np.zeros((canvas.shape[0] + 80, canvas.shape[1], 3), dtype=np.uint8)
        hud_canvas[:canvas.shape[0]] = canvas
        self._draw_hud(hud_canvas, frontier_count, agent_pos)

        cv2.imshow(window_name, hud_canvas)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self):
        return {
            "resolution": self.resolution,
            "grid_w":     self.grid_w,
            "grid_h":     self.grid_h,
            "origin_x":   self.origin_x,
            "origin_z":   self.origin_z,
            "grid":       self.grid.tolist(),
        }

    def save(self, path):
        vis = np.full((self.grid_h, self.grid_w), 128, dtype=np.uint8)
        vis[self.grid == FREE]     = 255
        vis[self.grid == OCCUPIED] = 0
        cv2.imwrite(path, vis)
        print(f"[OccupancyMap] Saved to {path}")