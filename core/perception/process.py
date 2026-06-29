# perception/process.py

import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
from scipy.spatial.transform import Rotation as R
import os
import time
from pathlib import Path

GRAPH_READ_ONLY = False

# Local imports
from perception.globals import (
    SALIENCY_MEAN_THRESHOLD,
    TARGET_CLASSES,
    SCENE_CLASS_NAMES,
    SCENE_CLASS_PROMPTS,
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

SAVE_CLIP_CROPS = True
CLIP_SAVE_DIR   = Path("classified_crops")
CLIP_SAVE_DIR.mkdir(exist_ok=True)


# ── Main processing entry point ───────────────────────────────────────────────

def process_message(json_str, update_vis=False, vis_params=None):

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

    # ── READ-ONLY FAST PATH ───────────────────────────────────────────────────
    # When the graph is pre-loaded and we are only navigating (not building),
    # skip every expensive pipeline stage: BASNet saliency, CLIP (object +
    # scene), EasyOCR, and occupancy map updates.  The sensor feed is still
    # decoded so that main.py can track the agent position from cam_pos.
    if GRAPH_READ_ONLY:
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

    # ── 5. Occupancy map ──────────────────────────────────────────────────────
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

    # ── 6. OCR ────────────────────────────────────────────────────────────────
    ocr_text = ocr.run_ocr(rgb_np)
    if ocr_text:
        print(f"[OCR] {ocr_text}")

    # ── 7. Saliency gate ──────────────────────────────────────────────────────
    try:
        sal_map  = saliency_detector.get_saliency_map(pil_img)
        sal_mean = float(np.mean(sal_map))
    except Exception as e:
        print(f"[Saliency] Error: {e}")
        _render_fallback(rgb_np, cam_pos, None, None, ocr_text)
        return

    if sal_mean < SALIENCY_MEAN_THRESHOLD:
        print(f"[Saliency] Too low ({sal_mean:.3f}) — skipping")
        _render_fallback(rgb_np, cam_pos, None, None, ocr_text)
        return

    # ── 8. Scene classification ───────────────────────────────────────────────
    scene_label = None
    scene_score = None

    if clip_model is not None:
        try:
            scene_prompt, scene_score = clip_model.top_score(pil_img, SCENE_CLASS_PROMPTS)
            if scene_prompt in SCENE_CLASS_PROMPTS:
                scene_label = SCENE_CLASS_NAMES[SCENE_CLASS_PROMPTS.index(scene_prompt)]
            else:
                scene_label = scene_prompt
            print(f"[SCENE] {scene_label} ({scene_score:.3f})")
        except Exception as e:
            print(f"[SCENE] Error: {e}")

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

    # ── 10. CLIP object classification on salient crop ────────────────────────
    clip_label = None
    clip_score = None
    clip_embed = np.zeros(512)

    if clip_model is not None and bbox is not None:
        x, y, w, h = bbox
        obj_crop = rgb_np[y:y+h, x:x+w]
        if obj_crop.size > 0:
            obj_pil = Image.fromarray(obj_crop)
            try:
                probs      = clip_model.score(obj_pil, TARGET_CLASSES)
                best_idx   = int(np.argmax(probs))
                clip_label = TARGET_CLASSES[best_idx]
                clip_score = float(probs[best_idx])
                print(f"[CLIP OBJ] {clip_label} ({clip_score:.3f})")
            except Exception as e:
                print(f"[CLIP OBJ] Error: {e}")

            try:
                emb_tensor = clip_model.encode_image(obj_pil)
                clip_embed = emb_tensor.cpu().detach().numpy().flatten()
            except Exception as e:
                print(f"[CLIP OBJ] Embedding error: {e}")

            if SAVE_CLIP_CROPS and clip_label is not None:
                _save_annotated_crop(obj_crop, clip_label, clip_score)

    # ── 11. Visualisation ─────────────────────────────────────────────────────
    _render(
        rgb_np       = rgb_np,
        depth_linear = depth_linear,
        sal_map      = sal_map,
        bbox         = bbox,
        pos_cam      = pos_cam,
        clip_label   = clip_label,
        clip_score   = clip_score,
        scene_label  = scene_label,
        scene_score  = scene_score,
        ocr_text     = ocr_text,
        cam_pos      = cam_pos,
    )

    # ── 12. World coordinate transform + landmark storage ─────────────────────
    if pos_cam is not None and distance is not None:
        point_cam_rh   = np.array([ pos_cam[0], -pos_cam[1], -pos_cam[2]])
        cam_pos_rh     = np.array([ cam_pos[0],  cam_pos[1], -cam_pos[2]])
        cam_rot_rh     = np.array([-cam_rot[0], -cam_rot[1],  cam_rot[2], cam_rot[3]])
        r              = R.from_quat(cam_rot_rh)
        point_world_rh = cam_pos_rh + r.apply(point_cam_rh)
        point_world    = np.array([point_world_rh[0],
                                   point_world_rh[1],
                                  -point_world_rh[2]])

        semantic_mapper.add_landmark(
            pos=point_world,
            clip_embedding=clip_embed,
            ocr_text=ocr_text,
            timestamp=timestamp,
            saliency_mean=sal_mean,
            clip_label=clip_label,
            scene_label=scene_label,
            scene_score=scene_score,
            distance=distance,
            clip_score=clip_score,
        )
        print(f"[Landmark] World pos: ({point_world[0]:.2f}, "
              f"{point_world[1]:.2f}, {point_world[2]:.2f})")
    else:
        print("[Landmark] No salient object or invalid depth — skipping")


# ── Private helpers ───────────────────────────────────────────────────────────

def _render(rgb_np, depth_linear, sal_map, bbox, pos_cam,
            clip_label, clip_score, scene_label, scene_score,
            ocr_text, cam_pos):
    draw_depth_map(depth_linear, "Depth Map")
    draw_saliency_map(sal_map, bbox, "Saliency")

    if bbox:
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        draw_rgb_with_bbox(rgb_np, bbox, (cx, cy), pos_cam, "RGB Detection",
                           clip_label=clip_label, clip_score=clip_score)
    else:
        disp = _scale_for_display(rgb_np)
        cv2.imshow("RGB Detection", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

    _render_info_overlay(rgb_np, scene_label, scene_score, ocr_text)
    occupancy_map.render(agent_pos=cam_pos, window_name="Occupancy Map")
    cv2.waitKey(1)


def _render_fallback(rgb_np, cam_pos, scene_label, scene_score, ocr_text):
    disp = _scale_for_display(rgb_np)
    cv2.imshow("RGB Detection", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
    _render_info_overlay(rgb_np, scene_label, scene_score, ocr_text)
    occupancy_map.render(agent_pos=cam_pos, window_name="Occupancy Map")
    cv2.waitKey(1)


def _render_info_overlay(rgb_np, scene_label, scene_score, ocr_text):
    disp = _scale_for_display(rgb_np.copy())
    disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    y_cursor = 30

    if scene_label is not None:
        scene_text = f"SCENE: {scene_label} ({scene_score*100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(scene_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(disp, (5, y_cursor - th - 4), (15 + tw, y_cursor + 4), (0, 0, 0), -1)
        cv2.putText(disp, scene_text, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y_cursor += 35

    if ocr_text:
        ocr_display = f"OCR: {ocr_text[:80]}"
        (tw, th), _ = cv2.getTextSize(ocr_display, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(disp, (5, y_cursor - th - 4), (15 + tw, y_cursor + 4), (0, 0, 0), -1)
        cv2.putText(disp, ocr_display, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)

    cv2.imshow("Scene / OCR", disp)


def _save_annotated_crop(obj_crop, clip_label, clip_score):
    try:
        ts         = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        safe_label = clip_label.replace("/", "_").replace(" ", "_")
        filename   = f"{ts}_{safe_label}_{clip_score:.3f}.png"
        filepath   = CLIP_SAVE_DIR / filename
        annotated  = obj_crop.copy()
        label_text = f"{clip_label} ({clip_score*100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (5, 5), (15 + tw, 25 + th), (0, 0, 0), -1)
        cv2.putText(annotated, label_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(filepath), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"[Crop] Saved → {filepath}")
    except Exception as e:
        print(f"[Crop] Save error: {e}")


def _scale_for_display(img_np, max_width=800):
    h, w = img_np.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img_np, None, fx=scale, fy=scale)
    return img_np