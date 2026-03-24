# world_points.py
import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # use non‑interactive backend for buffer rendering
import matplotlib.image as mpimg
import time
import torch

# Your existing perception modules
from perception.saliency import BASNetSaliency
import perception.clipy as clipy
import perception.ocr as ocr

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SALIENCY_MEAN_THRESHOLD = 0.04

TARGET_CLASSES = [
    "a long couch",
    "a small one-seat couch",
    "a king size bed",
    "big bookshelf",
    "long corridor",
    "corridor intersection",
    "room entrance",
    "poster",
    "sign"
]

# ----------------------------------------------------------------------
# Semantic Mapper Class
# ----------------------------------------------------------------------
class SemanticMapper:
    def __init__(self, spatial_threshold=1.0, semantic_threshold=0.85):
        """
        spatial_threshold: meters – if two positions are closer, consider merging
        semantic_threshold: cosine similarity – if CLIP embeddings are similar, consider merging
        """
        self.graph = nx.Graph()
        self.spatial_threshold = spatial_threshold
        self.semantic_threshold = semantic_threshold
        self.kd_tree = None
        self.kd_tree_ids = []
        self.last_landmark_id = None

    def _rebuild_kdtree(self):
        """Build a kd‑tree from all node positions."""
        if len(self.graph.nodes) == 0:
            self.kd_tree = None
            self.kd_tree_ids = []
            return
        positions = []
        ids = []
        for nid, data in self.graph.nodes(data=True):
            positions.append(data['position'])
            ids.append(nid)
        self.kd_tree = cKDTree(positions)
        self.kd_tree_ids = ids

    def _find_nearby(self, pos):
        """Return list of node IDs within spatial_threshold of pos."""
        if self.kd_tree is None:
            return []
        indices = self.kd_tree.query_ball_point(pos, self.spatial_threshold)
        return [self.kd_tree_ids[i] for i in indices]

    def add_landmark(self, pos, clip_embedding, ocr_text, timestamp, distance,
                     saliency_mean=None, clip_label=None, clip_score=None):

        # 1. Clean and Normalize Input Embedding
        if clip_embedding is not None:
            clip_embedding = np.array(clip_embedding).flatten()
            norm = np.linalg.norm(clip_embedding)
            if norm > 1e-6:
                clip_embedding /= norm
        else:
            # Avoid using zero-vectors in similarity calculations
            clip_embedding = None

        # 2. Candidate Selection via KD-Tree
        nearby_ids = self._find_nearby(pos)
        best_match = None
        best_sim = -1

        for nid in nearby_ids:
            node = self.graph.nodes[nid]
            existing_embed = np.array(node['clip_embed'])
            
            # 3. Semantic Similarity Check
            if clip_embedding is not None:
                sim = np.dot(clip_embedding, existing_embed)
                
                # Label Consistency Gate: Don't merge if labels are strongly contradictory
                if clip_label and node.get('clip_label'):
                    if clip_label != node['clip_label'] and sim < 0.95: 
                        continue 

                if sim > self.semantic_threshold and sim > best_sim:
                    best_sim = sim
                    best_match = nid

        # 4. Update or Create
        if best_match is not None:
            node = self.graph.nodes[best_match]
            count = node.get('obs_count', 1)

            # A. Weighted Position Update (Trust closer detections more)
            dist_weight = 1.0 / (1.0 + distance)
            pos_alpha = 0.3 * dist_weight 
            old_pos = np.array(node['position'])
            node['position'] = ((1 - pos_alpha) * old_pos + pos_alpha * np.array(pos)).tolist()

            # B. EMA Embedding Fusion
            if clip_embedding is not None:
                ema_alpha = 0.2
                node['clip_embed'] = (1 - ema_alpha) * np.array(node['clip_embed']) + ema_alpha * clip_embedding
                node['clip_embed'] /= np.linalg.norm(node['clip_embed']) # Re-normalize

            # C. Label Voting System
            if clip_label:
                votes = node.setdefault('label_votes', {node.get('clip_label', 'unknown'): count})
                votes[clip_label] = votes.get(clip_label, 0) + 1
                node['clip_label'] = max(votes, key=votes.get)

            # D. OCR Refinement
            if ocr_text and len(ocr_text) > len(node.get('ocr_text', '')):
                node['ocr_text'] = ocr_text

            node['obs_count'] = count + 1
            node['timestamp'] = timestamp
            landmark_id = best_match
            print(f"Updated landmark {landmark_id} (dist: {distance:.2f}m, votes: {node['label_votes']})")
        else:
            # Create new node
            landmark_id = f"lm_{len(self.graph.nodes)}"
            node_data = {
                'position': pos.tolist() if isinstance(pos, np.ndarray) else pos,
                'clip_embed': clip_embedding if clip_embedding is not None else np.zeros(512),
                'ocr_text': ocr_text or "",
                'timestamp': timestamp,
                'obs_count': 1,
                'clip_label': clip_label,
                'label_votes': {clip_label: 1} if clip_label else {},
                'saliency_mean': saliency_mean
            }
            self.graph.add_node(landmark_id, **node_data)
            print(f"Created new landmark {landmark_id} at {pos}")

        if self.last_landmark_id and self.last_landmark_id != landmark_id:
            self.graph.add_edge(self.last_landmark_id, landmark_id, relation="temporal")
        
        self.last_landmark_id = landmark_id
        self._rebuild_kdtree()
        return landmark_id


# ----------------------------------------------------------------------
# Global model instances (loaded once)
# ----------------------------------------------------------------------
saliency_detector = BASNetSaliency()
clip_model = clipy.CLIPModel()          # uses default "ViT-B/32"

# ----------------------------------------------------------------------
# Global storage for landmarks (now using SemanticMapper)
# ----------------------------------------------------------------------
semantic_mapper = SemanticMapper(spatial_threshold=1.0, semantic_threshold=0.85)

# ----------------------------------------------------------------------
# Drawing utilities (unchanged from your original code)
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

def draw_rgb_with_bbox(rgb_np, bbox, center, pos_3d, title="RGB"):
    img = rgb_np.copy()
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(img, center, 5, (0, 0, 255), -1)
    if pos_3d:
        text = f"X:{pos_3d[0]:.2f} Y:{pos_3d[1]:.2f} Z:{pos_3d[2]:.2f}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    h, w = img.shape[:2]
    if w > 800:
        scale = 800 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(title, img)

# ----------------------------------------------------------------------
# Core 3D projection (unchanged)
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

# ----------------------------------------------------------------------
# Main processing function (called from main.py for every message)
# ----------------------------------------------------------------------
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