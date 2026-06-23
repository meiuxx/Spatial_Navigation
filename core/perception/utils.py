import numpy as np
import cv2
from PIL import Image
import io

# ----------------------------------------------------------------------
# Drawing utilities
# ----------------------------------------------------------------------
def draw_depth_map(depth_map, title="Depth Map"):
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    h, w = depth_colored.shape[:2]
    if w < 400:
        scale = 400 / w
        depth_colored = cv2.resize(depth_colored, None, fx=scale, fy=scale)
    cv2.imshow(title, depth_colored)

def draw_saliency_map(sal_map, bbox=None, title="Saliency"):
    sal_uint8 = (sal_map * 255).astype(np.uint8)
    h, w = sal_uint8.shape
    scale = 400 / w
    disp = cv2.resize(sal_uint8, None, fx=scale, fy=scale)
    if bbox:
        x, y, w, h = bbox
        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)
        cv2.rectangle(disp, (x, y), (x + w, y + h), 255, 2)
    cv2.imshow(title, disp)

def draw_rgb_with_bbox(rgb_np, bbox, center, pos_3d, title="RGB",
                       clip_label=None, clip_score=None):
    """
    Draw RGB image with bounding box, 3D position, and optional CLIP label.
    """
    img = rgb_np.copy()
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(img, center, 5, (0, 0, 255), -1)
    if pos_3d:
        text = f"X:{pos_3d[0]:.2f} Y:{pos_3d[1]:.2f} Z:{pos_3d[2]:.2f}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw CLIP classification (if provided)
    if clip_label:
        label_text = f"CLIP: {clip_label} ({clip_score*100:.1f}%)" if clip_score else clip_label
        # Put text just above the bounding box (or inside if near top edge)
        text_y = y - 25 if y > 30 else y + h + 20
        cv2.putText(img, label_text, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

    h, w = img.shape[:2]
    if w > 800:
        scale = 800 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(title, img)

# ----------------------------------------------------------------------
# Core 3D projection
# ----------------------------------------------------------------------
def get_object_3d_position(rgb_image, saliency_map, depth_map, depth_w, depth_h, fov):
    h_img, w_img = rgb_image.shape[:2]

    sal_uint8 = (saliency_map * 255).astype(np.uint8)

    # Try Otsu thresholding
    _, binary = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mean_val = sal_uint8.mean()
        _, binary = cv2.threshold(sal_uint8, int(mean_val), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cx = x + w // 2
    cy = y + h // 2

    depth_x = int(cx * depth_w / w_img)
    depth_y = int(cy * depth_h / h_img)
    depth_x = np.clip(depth_x, 0, depth_w - 1)
    depth_y = np.clip(depth_y, 0, depth_h - 1)

    patch_size = 3
    half = patch_size // 2
    y_start = max(0, depth_y - half)
    y_end = min(depth_h, depth_y + half + 1)
    x_start = max(0, depth_x - half)
    x_end = min(depth_w, depth_x + half + 1)
    patch = depth_map[y_start:y_end, x_start:x_end]
    distance = np.median(patch)

    if distance <= 0 or distance > 100:
        return None, (x, y, w, h)

    fov_rad = np.radians(fov)
    fy = h_img / (2.0 * np.tan(fov_rad / 2.0))
    fx = fy  # square pixels assumed

    cx_img = w_img / 2.0
    cy_img = h_img / 2.0

    Z = distance
    X = (cx - cx_img) * Z / fx
    Y = (cy - cy_img) * Z / fy

    return (X, Y, Z), (x, y, w, h), distance

def compute_angle_to_object(bbox, image_width, fov_degrees):
    """Return horizontal angle (radians) from camera forward to object center."""
    fov_rad = np.radians(fov_degrees)
    cx = bbox[0] + bbox[2] // 2
    norm_x = (cx - image_width/2) / (image_width/2)
    angle = norm_x * (fov_rad / 2)
    return angle