# mapper.py
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

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