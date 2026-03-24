import base64
import numpy as np
from scipy.spatial.transform import Rotation as R

def decode_depth(depth_64, height, width):
    depth_bytes = base64.b64decode(depth_64)
    depth_flat = np.frombuffer(depth_bytes, dtype=np.float32)
    return depth_flat.reshape((height, width))

def sample_depth_at_pixel(depth_map, rgb_width, rgb_height, depth_w, depth_h, cx, cy, patch_size=3):
    # given certain pixel coordinates, we wanna find the equivalent
    # depth_map coordinates. The depth map is a mere downscaled ver of the rgb, same fov, same camera intrinsics
    dx=int(cx*depth_w/rgb_width)
    dy=int(cy*depth_h/rgb_height)
    dx = np.clip(dx, 0, depth_w -1)
    dy = np.clip(dy, 0, depth_h)

    #extract a small patch
    half = patch_size // 2
    y_start = max(0, dy - half)
    y_end = min(depth_h, dy+half+1)
    x_start = max(0, dx-half)
    x_end = min(depth_w, dx+half+1)
    patch = depth_map[y_start:y_end, x_start:x_end]
    
    if patch.size == 0:
        return None
    return np.median(patch)

def depth_to_world_point(cx, cy, distance, rgb_w, rgb_h, fov, cam_pos, cam_rot):
    # camera intrinsics
    fov_rad = np.radians(fov)
    fy= rgb_h / (2.0*np.tan(fov_rad/2.0))
    fx = fy #cause pixels are square so rgb_h=rgb_w
    cx_img = rgb_w / 2.0
    cy_img = rgb_h / 2.0

    # camera coordinates in unity using pinhole camera model equations
    Z = distance
    X = (cx - cx_img) * Z / fx
    Y = (cy-cy_img)*Z / fy
    point_cam = np.array([X, Y, Z])

    # transform to world coordinates (right handed)
    point_cam_rh = np.array([point_cam[0], -point_cam[1], -point_cam[2]])
    cam_pos_rh = np.array([cam_pos[0], cam_pos[1], -cam_pos[2]])
    cam_rot_rh = np.array([cam_rot[0], -cam_rot[1], cam_rot[2], cam_rot[3]])

    r=R.from_quat(cam_rot_rh)
    point_world_rh = cam_pos_rh + r.apply(point_cam_rh)

   #convert back to unity left-handed
    return np.array([point_world_rh[0], point_world_rh[1], -point_world_rh[2]]) 