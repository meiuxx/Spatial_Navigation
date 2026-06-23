# perception/process.py

import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path

# Local imports
from perception.globals import (
    SALIENCY_MEAN_THRESHOLD,
    TARGET_CLASSES,
    saliency_detector,
    clip_model,
    semantic_mapper,
)
from perception.utils import (
    draw_depth_map,
    draw_saliency_map,
    draw_rgb_with_bbox,
    get_object_3d_position,
)
import perception.ocr as ocr
from navigation.occupancy_map import OccupancyMap

# ── Shared occupancy map ──────────────────────────────────────────────────────

occupancy_map = OccupancyMap(
    resolution = 0.1,
    width_m    = 60.0,
    height_m   = 60.0,
    max_depth  = 15.0,
    col_step   = 4,
    row_step   = 4,
    floor_band = 1.2,
)

# ── Save classified crops ─────────────────────────────────────────────────────
SAVE_CLIP_CROPS = True                     # set to False to disable saving
CLIP_SAVE_DIR   = Path("classified_crops") # folder inside project root
CLIP_SAVE_DIR.mkdir(exist_ok=True)


# ── Main processing entry point ───────────────────────────────────────────────

def process_message(json_str, update_vis=False, vis_params=None):
    """
    Full per-frame perception pipeline:

      1.  Parse sensor JSON from Unity
      2.  Decode RGB + depth
      3.  Update occupancy map (every frame, unconditionally)
      4.  OCR on full frame
      5.  Saliency gate for landmark detection
      6.  3D position + bounding box from saliency mask
      7.  CLIP object classification on salient crop
      8.  World coordinate transform
      9.  Store landmark in semantic graph
      10. Visualisation
    """

    # ── 1. Parse JSON ─────────────────────────────────────────────────────────
    try:
        msg = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[Process] Invalid JSON: {e}")
        return

    # ── 2. Extract fields ─────────────────────────────────────────────────────
    try:
        rgb_b64   = msg['rgb']
        depth_b64 = msg['depth']
        depth_w   = msg['depth_width']
        depth_h   = msg['depth_height']
        rgb_w     = msg['rgb_width']
        rgb_h     = msg['rgb_height']
        fov       = msg['fov']
        timestamp = msg.get('timestamp', 0)
        cam_pos   = np.array([msg['cam_pos_x'], msg['cam_pos_y'], msg['cam_pos_z']])
        cam_rot   = np.array([msg['cam_rot_x'], msg['cam_rot_y'],
                              msg['cam_rot_z'], msg['cam_rot_w']])
    except KeyError as e:
        print(f"[Process] Missing key: {e}")
        return

    # ── 3. Decode RGB ─────────────────────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(base64.b64decode(rgb_b64))).convert('RGB')
        rgb_np  = np.array(pil_img)
    except Exception as e:
        print(f"[Process] RGB decode error: {e}")
        return

    # ── 4. Decode depth ───────────────────────────────────────────────────────
    try:
        depth_flat = np.frombuffer(base64.b64decode(depth_b64), dtype=np.float32)
        if len(depth_flat) != depth_w * depth_h:
            print("[Process] Depth size mismatch")
            return
        depth_linear = depth_flat.reshape((depth_h, depth_w))
    except Exception as e:
        print(f"[Process] Depth decode error: {e}")
        return

    # ── 5. Occupancy map (unconditional — every frame) ────────────────────────
    occupancy_map.update(
        depth_linear = depth_linear,
        depth_w      = depth_w,
        depth_h      = depth_h,
        rgb_w        = rgb_w,
        rgb_h        = rgb_h,
        fov          = fov,
        cam_pos      = cam_pos,
        cam_rot      = cam_rot,
    )
    print(f"[Occupancy] {occupancy_map.get_frontier_count()} frontiers")

    # ── 6. OCR (full frame, every frame) ──────────────────────────────────────
    ocr_text = ocr.run_ocr(rgb_np)
    if ocr_text:
        print(f"[OCR] {ocr_text}")

    # ── 7. Saliency gate ──────────────────────────────────────────────────────
    try:
        sal_map  = saliency_detector.get_saliency_map(pil_img)
        sal_mean = float(np.mean(sal_map))
    except Exception as e:
        print(f"[Saliency] Error: {e}")
        occupancy_map.render(agent_pos=cam_pos, window_name="Occupancy Map")
        return

    if sal_mean < SALIENCY_MEAN_THRESHOLD:
        print(f"[Saliency] Too low ({sal_mean:.3f}) — skipping landmark detection")
        occupancy_map.render(agent_pos=cam_pos, window_name="Occupancy Map")
        cv2.waitKey(1)
        return

    # ── 9. 3D position + bounding box from saliency ───────────────────────────
    result = get_object_3d_position(
        rgb_np, sal_map, depth_linear, depth_w, depth_h, fov
    )

    if result is None or len(result) < 2:
        pos_cam, bbox, distance = None, None, None
    elif len(result) == 2:
        pos_cam, bbox, distance = None, result[1], None
    else:
        pos_cam, bbox, distance = result

    # ── 10. CLIP object classification on salient crop ─────────────────────────
    clip_label = None
    clip_score = None
    clip_embed = np.zeros(512)

    if clip_model is not None and bbox is not None:
        x, y, w, h = bbox
        obj_crop   = rgb_np[y:y+h, x:x+w]
        if obj_crop.size > 0:
            obj_pil    = Image.fromarray(obj_crop)
            probs      = clip_model.score(obj_pil, TARGET_CLASSES)
            best_idx   = int(np.argmax(probs))
            clip_label = TARGET_CLASSES[best_idx]
            clip_score = float(probs[best_idx])
            try:
                emb_tensor = clip_model.encode_image(obj_pil)
                clip_embed = emb_tensor.cpu().detach().numpy().flatten()
            except Exception as e:
                print(f"[CLIP] Embedding error: {e}")
            print(f"[CLIP] {clip_label} ({clip_score:.3f})")

                        # ── Save annotated cropped image ─────────────────────────────────────
            if SAVE_CLIP_CROPS and obj_crop.size > 0:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                safe_label = clip_label.replace("/", "_").replace(" ", "_")
                filename = f"{timestamp}_{safe_label}_{clip_score:.3f}_annotated.png"
                filepath = CLIP_SAVE_DIR / filename

                # Draw label on a copy of the crop
                annotated_crop = obj_crop.copy()
                label_text = f"{clip_label} ({clip_score*100:.1f}%)"
                # Draw a background rectangle for better readability
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_crop, (5, 5), (15 + text_w, 25 + text_h), (0, 0, 0), -1)
                cv2.putText(annotated_crop, label_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imwrite(str(filepath), cv2.cvtColor(annotated_crop, cv2.COLOR_RGB2BGR))
                print(f"[Saved annotated crop] {filepath}")
                
    # ── 10. Visualisation ─────────────────────────────────────────────────────
    draw_depth_map(depth_linear, "Depth Map")
    draw_saliency_map(sal_map, bbox, "Saliency")

    if bbox:
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        draw_rgb_with_bbox(rgb_np, bbox, (cx, cy), pos_cam, "RGB Detection",
                           clip_label=clip_label, clip_score=clip_score)
    else:
        disp = cv2.resize(rgb_np, None, fx=0.5, fy=0.5) \
               if rgb_np.shape[1] > 800 else rgb_np.copy()
        cv2.imshow("RGB Detection", disp)

    # Overlay OCR text on preview
    disp_info = rgb_np.copy()
    if rgb_np.shape[1] > 800:
        disp_info = cv2.resize(disp_info, None, fx=0.5, fy=0.5)
    if ocr_text:
        cv2.putText(disp_info, f"OCR: {ocr_text[:60]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow("OCR", disp_info)

    occupancy_map.render(agent_pos=cam_pos, window_name="Occupancy Map")
    cv2.waitKey(1)

    # ── 11. World coordinate transform + landmark storage ─────────────────────
    if pos_cam is not None and distance is not None:
        point_cam_rh   = np.array([ pos_cam[0], -pos_cam[1], -pos_cam[2]])
        cam_pos_rh     = np.array([ cam_pos[0],  cam_pos[1], -cam_pos[2]])
        cam_rot_rh     = np.array([-cam_rot[0], -cam_rot[1],  cam_rot[2], cam_rot[3]])
        r              = R.from_quat(cam_rot_rh)
        point_world_rh = cam_pos_rh + r.apply(point_cam_rh)
        point_world    = np.array([point_world_rh[0],
                                   point_world_rh[1],
                                   -point_world_rh[2]])

        print(f"[Landmark] World pos: ({point_world[0]:.2f}, "
              f"{point_world[1]:.2f}, {point_world[2]:.2f})")

        semantic_mapper.add_landmark(
            pos            = point_world,
            clip_embedding = clip_embed,
            ocr_text       = ocr_text,
            timestamp      = timestamp,
            saliency_mean  = sal_mean,
            clip_label     = clip_label,
            distance       = distance,
            clip_score     = clip_score,
        )
    else:
        print("[Landmark] No salient object or invalid depth — skipping")