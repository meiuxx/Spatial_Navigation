import base64
import numpy as np
from scipy.spatial.transform import Rotation as R


def decode_depth(depth_64, height, width):
    depth_bytes = base64.b64decode(depth_64)
    depth_flat = np.frombuffer(depth_bytes, dtype=np.float32)
    return depth_flat.reshape((height, width))


def sample_depth_at_pixel(depth_map, rgb_width, rgb_height, depth_w, depth_h, cx, cy, patch_size=3):
    """
    Map an RGB pixel coordinate (cx, cy) to its equivalent position in the
    depth map (which is a downscaled version of the RGB frame, same FOV and
    camera intrinsics) and return the median depth of a small patch around it.
    """
    dx = int(cx * depth_w / rgb_width)
    dy = int(cy * depth_h / rgb_height)
    # FIX: was np.clip(dy, 0, depth_h) — depth_h is out-of-bounds, must be depth_h - 1
    dx = np.clip(dx, 0, depth_w - 1)
    dy = np.clip(dy, 0, depth_h - 1)

    # Extract a small patch around the mapped pixel
    half = patch_size // 2
    y_start = max(0, dy - half)
    y_end   = min(depth_h, dy + half + 1)
    x_start = max(0, dx - half)
    x_end   = min(depth_w, dx + half + 1)
    patch = depth_map[y_start:y_end, x_start:x_end]

    if patch.size == 0:
        return None
    return float(np.median(patch))


def depth_to_world_point(cx, cy, distance, rgb_w, rgb_h, fov, cam_pos, cam_rot):
    """
    Unproject a 2-D image point (cx, cy) at a known metric distance into a
    3-D world coordinate, accounting for Unity's left-handed coordinate system.

    cam_pos : (x, y, z)  — Unity world position
    cam_rot : (x, y, z, w) — Unity quaternion (left-handed)
    Returns  : np.array([x, y, z]) in Unity world space
    """
    # ── Camera intrinsics (pinhole model, square pixels) ──────────────────
    fov_rad  = np.radians(fov)
    fy       = rgb_h / (2.0 * np.tan(fov_rad / 2.0))
    fx       = fy                    # square pixels → fx == fy
    cx_img   = rgb_w / 2.0
    cy_img   = rgb_h / 2.0

    # ── Camera-space 3-D point (Unity convention: +X right, +Y up, +Z forward)
    Z = distance
    X = (cx - cx_img) * Z / fx
    Y = (cy - cy_img) * Z / fy
    point_cam = np.array([X, Y, Z])

    # ── Convert Unity left-handed → right-handed for scipy rotation ───────
    #    Unity LH: +X right, +Y up, +Z forward
    #    RH equiv : +X right, +Y up, +Z backward  → flip Z
    point_cam_rh = np.array([ point_cam[0], -point_cam[1], -point_cam[2]])
    cam_pos_rh   = np.array([ cam_pos[0],    cam_pos[1],   -cam_pos[2]])
    # Unity quaternion (x, y, z, w) LH → RH: negate x and z (or equivalently negate y)
    cam_rot_rh   = np.array([ cam_rot[0], -cam_rot[1],  cam_rot[2], cam_rot[3]])

    # ── Rotate and translate ──────────────────────────────────────────────
    r = R.from_quat(cam_rot_rh)          # scipy expects (x, y, z, w)
    point_world_rh = cam_pos_rh + r.apply(point_cam_rh)

    # ── Convert back to Unity left-handed (flip Z) ────────────────────────
    return np.array([point_world_rh[0], point_world_rh[1], -point_world_rh[2]])