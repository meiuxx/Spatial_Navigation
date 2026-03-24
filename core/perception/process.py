# process.py
import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
from scipy.spatial.transform import Rotation as R

# Local imports
from perception.globals import SALIENCY_MEAN_THRESHOLD, TARGET_CLASSES
from perception.utils import (
    draw_depth_map,
    draw_saliency_map,
    draw_rgb_with_bbox,
    get_object_3d_position
)
from perception.globals import saliency_detector, clip_model, semantic_mapper
import perception.ocr as ocr   # also needed for OCR inside process_message

def process_message(json_str, update_vis=False, vis_params=None):
    """
    Parse JSON, run saliency + CLIP, compute 3D position, store in graph.
    If update_vis is True and vis_params are provided, also return an image of the graph.
    """
    try:
        msg = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return

    # Extract fields
    try:
        rgb_b64 = msg['rgb']
        depth_b64 = msg['depth']
        depth_w = msg['depth_width']
        depth_h = msg['depth_height']
        rgb_w = msg['rgb_width']
        rgb_h = msg['rgb_height']
        fov = msg['fov']
        timestamp = msg.get('timestamp', 0)
        cam_pos = np.array([msg['cam_pos_x'], msg['cam_pos_y'], msg['cam_pos_z']])
        cam_rot = np.array([msg['cam_rot_x'], msg['cam_rot_y'], msg['cam_rot_z'], msg['cam_rot_w']])
    except KeyError as e:
        print(f"Missing key: {e}")
        return

    # Decode RGB
    try:
        rgb_bytes = base64.b64decode(rgb_b64)
        pil_img = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
        rgb_np = np.array(pil_img)
        ocr_text = ocr.run_ocr(rgb_np)
        print(f"detected text: {ocr_text}")
    except Exception as e:
        print(f"RGB decode error: {e}")
        return

    # Decode depth
    try:
        depth_bytes = base64.b64decode(depth_b64)
        depth_flat = np.frombuffer(depth_bytes, dtype=np.float32)
        if len(depth_flat) != depth_w * depth_h:
            print("Depth size mismatch")
            return
        depth_linear = depth_flat.reshape((depth_h, depth_w))
    except Exception as e:
        print(f"Depth decode error: {e}")
        return

    # Saliency
    try:
        sal_map = saliency_detector.get_saliency_map(pil_img)
    except Exception as e:
        print(f"Saliency error: {e}")
        return

    # Saliency threshold filter
    sal_mean = np.mean(sal_map)
    if sal_mean < SALIENCY_MEAN_THRESHOLD:
        print(f"Saliency too low ({sal_mean:.3f} < {SALIENCY_MEAN_THRESHOLD}) – skipping frame")
        return

    # 3D position and bounding box
    pos_cam, bbox, distance = get_object_3d_position(rgb_np, sal_map, depth_linear, depth_w, depth_h, fov)

    # CLIP on the cropped object
    clip_label = None
    clip_score = None
    clip_embed = None
    if clip_model is not None and bbox is not None:
        x, y, w, h = bbox
        obj_crop = rgb_np[y:y+h, x:x+w]
        if obj_crop.size > 0:
            obj_pil = Image.fromarray(obj_crop)
            probs = clip_model.score(obj_pil, TARGET_CLASSES)
            best_idx = int(np.argmax(probs))
            clip_label = TARGET_CLASSES[best_idx]
            clip_score = probs[best_idx]
            # Get embedding (assumes clip_model.encode_image exists)
            try:
                emb_tensor = clip_model.encode_image(obj_pil)
                clip_embed = emb_tensor.cpu().detach().numpy().flatten()
            except Exception as e:
                print(f"Could not get embedding: {e}")
                clip_embed = np.zeros(512)  # fallback
            print(f"CLIP top class: {clip_label} ({clip_score:.3f})")
        else:
            clip_embed = np.zeros(512)
    else:
        clip_embed = np.zeros(512)

    # Visualisation (unchanged)
    draw_depth_map(depth_linear, "Depth Map")
    draw_saliency_map(sal_map, bbox, "Saliency")
    if bbox:
        cx = bbox[0] + bbox[2]//2
        cy = bbox[1] + bbox[3]//2
        draw_rgb_with_bbox(rgb_np, bbox, (cx, cy), pos_cam, "RGB Detection")
    else:
        img = cv2.resize(rgb_np, None, fx=0.5, fy=0.5) if rgb_np.shape[1] > 800 else rgb_np
        cv2.imshow("RGB Detection", img)
    cv2.waitKey(1)

    # World coordinate transformation
    if pos_cam is not None:
        point_cam_rh = np.array([pos_cam[0], -pos_cam[1], -pos_cam[2]])
        cam_pos_rh = np.array([cam_pos[0], cam_pos[1], -cam_pos[2]])
        cam_rot_rh = np.array([-cam_rot[0], -cam_rot[1], cam_rot[2], cam_rot[3]])
        r = R.from_quat(cam_rot_rh)
        point_world_rh = cam_pos_rh + r.apply(point_cam_rh)
        point_world = np.array([point_world_rh[0], point_world_rh[1], -point_world_rh[2]])

        print(f"World landmark: ({point_world[0]:.2f}, {point_world[1]:.2f}, {point_world[2]:.2f})")

        # Add to semantic mapper
        semantic_mapper.add_landmark(
            pos=point_world,
            clip_embedding=clip_embed,
            ocr_text=ocr_text,
            timestamp=timestamp,
            saliency_mean=sal_mean,
            clip_label=clip_label,
            distance=distance,
            clip_score=clip_score
        )
    else:
        print("No salient object detected or invalid depth")