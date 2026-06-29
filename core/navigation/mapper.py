# mapper.py
import json
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

# ── Tuning knobs (all in one place) ──────────────────────────────────────────

SEARCH_RADIUS_M        = 3.0
CLOSE_RANGE_M          = 2.5
CLOSE_RANGE_SIM_FLOOR  = 0.50
SEMANTIC_THRESHOLD     = 0.72
SPATIAL_FALLBACK_DIST  = 1.5
SPATIAL_FALLBACK_SIM   = 0.55
MERGE_SIM_THRESHOLD    = 0.88
MERGE_DIST_THRESHOLD   = 1.2
EMA_BASE_ALPHA         = 0.15
EMA_MAX_ALPHA          = 0.40
POS_ALPHA_BASE         = 0.25
POS_ALPHA_MAX          = 0.45

# ── Spatial scene consensus ──────────────────────────────────────────────────
# Landmarks within this distance of each other are treated as "same room"
# when pooling scene_label_votes for consensus. Too small and a large real
# room won't form one cluster; too large and two adjacent different rooms
# merge into one. Tune against your actual level's room sizes.
CLUSTER_RADIUS_M       = 5.0

# Room types that physically exist only once in this level. If more than
# one spatial cluster claims one of these after consensus voting, only the
# cluster with the larger total vote mass keeps it -- the rest fall back to
# their next-best label. Don't include types that legitimately recur
# (patient room, corridor, bathroom, examination, patient rooms hallway,
# etc.) -- adjust this set to match your actual hospital layout.
SINGLE_INSTANCE_SCENES = {
    "Administration", "conference room", "emergency", "imaging", "lab",
    "lobby", "main entrance", "OR_area", "OR_sterilization", "pharamacy",
    "reception-nurse station", "supply room", "VIP waiting area",
    "administration lounge", "cafeteria",
}

# Minimum scene_score (from CLIP full-frame classifier) for a scene vote to
# carry any weight.  Observations below this are discarded rather than adding
# noise to the majority vote.
SCENE_MIN_CONFIDENCE   = 0.25

# How much weight a scene vote gets = scene_score ** SCENE_VOTE_POWER.
# Power > 1 suppresses low-confidence votes more aggressively.
SCENE_VOTE_POWER       = 2.0

# Object label → plausible scene labels.
# When the object-level CLIP label strongly implies a room type, that signal
# is added as a synthetic scene vote weighted by OBJECT_SCENE_PRIOR_WEIGHT.
# This cross-checks the full-frame scene classifier against what was actually
# detected in the salient crop — catches cases where the frame looks like a
# corridor but the salient crop is clearly a surgical table.
OBJECT_SCENE_PRIOR_WEIGHT = 1.5   # synthetic vote weight relative to one obs

OBJECT_TO_SCENE_PRIOR = {
    # Surgical / OR
    "a surgical operating table":                    ["OR_area"],
    "a surgical ceiling light":                      ["OR_area"],
    "a surgical instrument tray":                    ["OR_area"],
    "a surgical equipment trolley":                  ["OR_area", "OR_sterilization"],
    "an anesthesia machine":                         ["OR_area"],
    "a ventilator machine":                          ["OR_area"],
    "an anesthesia/ventilator cart with vital signs monitor": ["OR_area"],
    # Sterilisation
    "OR_sterilization":                              ["OR_sterilization"],
    # Imaging
    "a ultrasound machine":                          ["imaging"],
    "an MRI machine":                                ["imaging"],
    "a medical X-ray viewer lightbox":               ["imaging"],
    # Lab
    "a microscope":                                  ["lab"],
    "a specimen tube rack":                          ["lab"],
    "a blue laboratory analysis machine":            ["lab"],
    "a laboratory sample containers rack":           ["lab"],
    "a laboratory conical flask":                    ["lab"],
    "a laboratory workbench":                        ["lab"],
    # Pharmacy
    "a retail shelf unit with price tags":           ["pharamacy"],
    "a pegboard display wall":                       ["pharamacy"],
    "a chest freezer display unit":                  ["pharamacy", "dining_lounge"],
    # Patient room
    "a hospital bed":                                ["patient room", "patient rooms hallway"],
    "a wooden wardrobe":                             ["patient room", "patient rooms hallway"],
    "a headwall panel":                              ["patient room"],
    # Examination
    "a medical examination table":                   ["examination"],
    "an IV pole":                                    ["examination", "patient room"],
    "a green medical screen divider":                ["examination", "patient room"],
    # Reception / lobby
    "a hospital reception desk":                     ["reception-nurse station", "lobby"],
    "a wheelchair":                                  ["lobby", "main entrance"],
    "a padded waiting area chair":                   ["lobby", "main entrance"],
    "a long couch or sofa":                          ["lobby"],
    # Cafeteria
    "a cafeteria dining table with attached seating": ["dining_lounge"],
    "a vending machine with snacks and drinks":      ["dining_lounge"],
    "a high bar stool chair":                        ["dining_lounge"],
    # Bathroom
    "a toilet":                                      ["bathroom"],
    "a multi basin sink":                            ["bathroom"],
    "a countertop washbasin":                        ["bathroom"],
    "a bathroom mirror":                             ["bathroom"],
    "a soap dispenser":                              ["bathroom"],
    # Supply room
    "a multi tier supply shelving unit":             ["supply room"],
    # Conference / admin
    "a meeting room table":                          ["conference room"],
    "a meeting room chair":                          ["conference room"],
    "a large office desk":                           ["Administration", "examination"],
    # Emergency
    "an emergency department sign in red font":      ["emergency"],
}


def _initial_scene_votes(scene_label, scene_score, clip_label, clip_score):
    """
    Build the initial scene_label_votes dict for a newly created landmark node,
    using the same confidence-weighted + object-prior logic as _fuse_observation.
    """
    votes = {}
    if scene_label and scene_score is not None and scene_score >= SCENE_MIN_CONFIDENCE:
        votes[scene_label] = float(scene_score) ** SCENE_VOTE_POWER
    elif scene_label:
        votes[scene_label] = 0.0  # record the label but give it zero weight

    if clip_label and clip_label in OBJECT_TO_SCENE_PRIOR:
        implied   = OBJECT_TO_SCENE_PRIOR[clip_label]
        obj_conf  = float(clip_score) if clip_score is not None else 0.5
        prior_v   = OBJECT_SCENE_PRIOR_WEIGHT * obj_conf / len(implied)
        for s in implied:
            votes[s] = votes.get(s, 0.0) + prior_v

    return votes


class SemanticMapper:
    """
    Maintains a spatial-semantic graph of observed landmarks.
    Each node stores both an object-level CLIP label (from the salient crop)
    and a scene-level label (from the full-frame fine-tuned classifier).
    """

    def __init__(
        self,
        spatial_threshold : float = SEMANTIC_THRESHOLD,
        semantic_threshold: float = SEMANTIC_THRESHOLD,
    ):
        self.graph              = nx.Graph()
        self.spatial_threshold  = spatial_threshold
        self.semantic_threshold = semantic_threshold
        self.kd_tree            = None
        self.kd_tree_ids        = []
        self.last_landmark_id   = None
        self._obs_since_merge   = 0
        self._next_lm_id        = 0

    # ── KD-tree ───────────────────────────────────────────────────────────────

    def _rebuild_kdtree(self):
        if len(self.graph.nodes) == 0:
            self.kd_tree     = None
            self.kd_tree_ids = []
            return
        positions, ids = [], []
        for nid, data in self.graph.nodes(data=True):
            positions.append(data['position'])
            ids.append(nid)
        self.kd_tree     = cKDTree(positions)
        self.kd_tree_ids = ids

    def _find_nearby(self, pos, radius=None):
        if self.kd_tree is None:
            return []
        r       = radius if radius is not None else SEARCH_RADIUS_M
        indices = self.kd_tree.query_ball_point(pos, r)
        return [self.kd_tree_ids[i] for i in indices]

    # ── Candidate scoring ─────────────────────────────────────────────────────

    def _score_candidate(self, nid, pos, clip_embedding, clip_label, clip_score):
        node     = self.graph.nodes[nid]
        node_pos = np.array(node['position'])
        dist_3d  = float(np.linalg.norm(np.array(pos) - node_pos))

        existing_embed = np.array(node['clip_embed'])
        e_norm         = np.linalg.norm(existing_embed)
        if e_norm < 1e-6:
            return None, None

        if clip_embedding is None:
            if dist_3d < SPATIAL_FALLBACK_DIST:
                return 0.60, 0.60
            return None, None

        sim = float(np.dot(clip_embedding, existing_embed / e_norm))

        penalty = 0.0
        if clip_label and node.get('clip_label'):
            if clip_label != node['clip_label']:
                n_obs      = node.get('obs_count', 1)
                label_conf = min(1.0, n_obs / 5.0)
                penalty    = 0.08 * label_conf

        proximity_bonus = max(
            0.0,
            (SPATIAL_FALLBACK_DIST - dist_3d) / SPATIAL_FALLBACK_DIST
        ) * 0.06

        score = sim - penalty + proximity_bonus
        return score, sim

    # ── Match selection ───────────────────────────────────────────────────────

    def _find_best_match(self, pos, clip_embedding, clip_label, clip_score):
        nearby_ids = self._find_nearby(pos, radius=SEARCH_RADIUS_M)
        if not nearby_ids:
            return None

        pos_arr     = np.array(pos)
        best_id     = None
        best_score  = -np.inf
        close_id    = None
        close_sim   = -np.inf
        spatial_id  = None
        spatial_sim = -np.inf

        for nid in nearby_ids:
            score, sim = self._score_candidate(
                nid, pos, clip_embedding, clip_label, clip_score
            )
            if score is None:
                continue

            node_pos = np.array(self.graph.nodes[nid]['position'])
            dist_3d  = float(np.linalg.norm(pos_arr - node_pos))

            if score > self.semantic_threshold and score > best_score:
                best_score = score
                best_id    = nid

            if (dist_3d < CLOSE_RANGE_M
                    and sim > CLOSE_RANGE_SIM_FLOOR
                    and sim > close_sim):
                close_sim = sim
                close_id  = nid

            if (dist_3d < SPATIAL_FALLBACK_DIST
                    and sim > 0.0
                    and sim > spatial_sim):
                spatial_sim = sim
                spatial_id  = nid

        if best_id    is not None: return best_id
        if close_id   is not None: return close_id
        if spatial_id is not None: return spatial_id
        return None

    # ── Observation fusion ────────────────────────────────────────────────────

    def _fuse_observation(self, nid, pos, clip_embedding, clip_label,
                          clip_score, ocr_text, timestamp, distance,
                          saliency_mean, scene_label, scene_score=None):
        node  = self.graph.nodes[nid]
        count = node.get('obs_count', 1)

        # A. Position update weighted by distance
        dist_weight = 1.0 / (1.0 + max(distance, 0.1))
        pos_alpha   = np.clip(
            POS_ALPHA_BASE + dist_weight * 0.2, POS_ALPHA_BASE, POS_ALPHA_MAX
        )
        old_pos          = np.array(node['position'])
        node['position'] = (
            (1 - pos_alpha) * old_pos + pos_alpha * np.array(pos)
        ).tolist()

        # B. Confidence-weighted EMA embedding update
        if clip_embedding is not None:
            confidence = float(clip_score) if clip_score is not None else (
                min(1.0, saliency_mean * 10) if saliency_mean else 0.5
            )
            ema_alpha = np.clip(
                EMA_BASE_ALPHA + confidence * (EMA_MAX_ALPHA - EMA_BASE_ALPHA),
                EMA_BASE_ALPHA, EMA_MAX_ALPHA,
            )
            old_embed        = np.array(node['clip_embed'])
            merged           = (1 - ema_alpha) * old_embed + ema_alpha * clip_embedding
            norm             = np.linalg.norm(merged)
            node['clip_embed'] = (merged / norm if norm > 1e-6 else merged).tolist()

        # C. Score-weighted object label voting
        if clip_label:
            vote_weight = max(1, round((clip_score or 0.5) * 3))
            votes       = node.setdefault(
                'label_votes',
                {node.get('clip_label', 'unknown'): count}
            )
            votes[clip_label]  = votes.get(clip_label, 0) + vote_weight
            node['clip_label'] = max(votes, key=votes.get)

        # D. Scene label voting — confidence-weighted + object-prior fusion
        #
        # Three signals are combined into scene_label_votes (float weights):
        #
        #   D1. Full-frame CLIP scene score (confidence-weighted, power-scaled).
        #       Low-confidence frames (below SCENE_MIN_CONFIDENCE) are discarded
        #       entirely rather than adding noise.
        #
        #   D2. Object→scene prior: if the object-level CLIP label strongly
        #       implies a room type (e.g. "surgical operating table" → OR_area),
        #       inject synthetic votes for those scene labels.  This cross-checks
        #       the full-frame classifier against what was actually in the crop —
        #       critical for transitional views (doorways, corridor boundaries)
        #       where the full frame looks ambiguous but the salient object is
        #       unambiguous.
        #
        #   D3. OCR boost: if OCR text was detected in this frame, slightly
        #       upweight whichever scene label just won D1 (signage is reliable).
        scene_votes = node.setdefault('scene_label_votes', {})

        if scene_label and scene_score is not None and scene_score >= SCENE_MIN_CONFIDENCE:
            # D1: confidence-weighted vote
            vote = (float(scene_score) ** SCENE_VOTE_POWER)
            # D3: OCR boost — signage makes the scene classification more reliable
            if ocr_text and len(ocr_text.strip()) > 2:
                vote *= 1.3
            scene_votes[scene_label] = scene_votes.get(scene_label, 0.0) + vote

        # D2: object→scene prior
        if clip_label and clip_label in OBJECT_TO_SCENE_PRIOR:
            implied_scenes = OBJECT_TO_SCENE_PRIOR[clip_label]
            # Weight by object confidence; split evenly across implied scenes
            obj_conf   = float(clip_score) if clip_score is not None else 0.5
            prior_vote = OBJECT_SCENE_PRIOR_WEIGHT * obj_conf / len(implied_scenes)
            for implied in implied_scenes:
                scene_votes[implied] = scene_votes.get(implied, 0.0) + prior_vote

        if scene_votes:
            node['scene_label'] = max(scene_votes, key=scene_votes.get)

        # E. OCR: keep longest string
        if ocr_text and len(ocr_text) > len(node.get('ocr_text', '')):
            node['ocr_text'] = ocr_text

        # F. Track best saliency
        if saliency_mean is not None:
            node['best_saliency'] = max(
                node.get('best_saliency', 0.0), saliency_mean
            )

        node['obs_count'] = count + 1
        node['timestamp'] = timestamp

        print(
            f"[Mapper] Updated {nid} | obs={node['obs_count']} "
            f"dist={distance:.2f}m | obj={node['clip_label']} "
            f"scene={node.get('scene_label')} "
            f"votes={node.get('label_votes', {})}"
        )

    # ── Duplicate merging ─────────────────────────────────────────────────────

    def _merge_duplicates(self):
        ids = list(self.graph.nodes)
        if len(ids) < 2:
            return

        positions = np.array([self.graph.nodes[i]['position'] for i in ids])
        tree      = cKDTree(positions)
        merged    = set()
        changed   = False

        for i, nid_a in enumerate(ids):
            if nid_a in merged:
                continue
            node_a  = self.graph.nodes[nid_a]
            embed_a = np.array(node_a['clip_embed'])
            norm_a  = np.linalg.norm(embed_a)
            if norm_a < 1e-6:
                continue

            indices = tree.query_ball_point(positions[i], MERGE_DIST_THRESHOLD)
            for j in indices:
                nid_b = ids[j]
                if nid_b == nid_a or nid_b in merged:
                    continue
                node_b  = self.graph.nodes[nid_b]
                embed_b = np.array(node_b['clip_embed'])
                norm_b  = np.linalg.norm(embed_b)
                if norm_b < 1e-6:
                    continue

                sim = float(np.dot(embed_a / norm_a, embed_b / norm_b))
                if sim < MERGE_SIM_THRESHOLD:
                    continue

                keep, drop = (
                    (nid_a, nid_b)
                    if node_a.get('obs_count', 1) >= node_b.get('obs_count', 1)
                    else (nid_b, nid_a)
                )
                self._absorb(keep, drop)
                merged.add(drop)
                changed = True
                print(f"[Mapper] Merged duplicate {drop} → {keep} (sim={sim:.3f})")

        if changed:
            self._rebuild_kdtree()

    def _absorb(self, keep_id, drop_id):
        keep = self.graph.nodes[keep_id]
        drop = self.graph.nodes[drop_id]

        kc    = keep.get('obs_count', 1)
        dc    = drop.get('obs_count', 1)
        total = kc + dc

        # Position
        keep['position'] = (
            (np.array(keep['position']) * kc + np.array(drop['position']) * dc) / total
        ).tolist()

        # Embedding
        merged = (
            np.array(keep['clip_embed']) * kc + np.array(drop['clip_embed']) * dc
        ) / total
        norm = np.linalg.norm(merged)
        keep['clip_embed'] = (merged / norm if norm > 1e-6 else merged).tolist()

        # Object label votes
        for label, votes in drop.get('label_votes', {}).items():
            keep.setdefault('label_votes', {})[label] = (
                keep.get('label_votes', {}).get(label, 0) + votes
            )
        if keep.get('label_votes'):
            keep['clip_label'] = max(keep['label_votes'], key=keep['label_votes'].get)

        # Scene label votes
        for label, votes in drop.get('scene_label_votes', {}).items():
            keep.setdefault('scene_label_votes', {})[label] = (
                keep.get('scene_label_votes', {}).get(label, 0) + votes
            )
        if keep.get('scene_label_votes'):
            keep['scene_label'] = max(
                keep['scene_label_votes'], key=keep['scene_label_votes'].get
            )

        # OCR
        if len(drop.get('ocr_text', '')) > len(keep.get('ocr_text', '')):
            keep['ocr_text'] = drop['ocr_text']

        keep['obs_count']     = total
        keep['best_saliency'] = max(
            keep.get('best_saliency', 0.0), drop.get('best_saliency', 0.0)
        )

        for nbr in list(self.graph.neighbors(drop_id)):
            if nbr != keep_id:
                self.graph.add_edge(
                    keep_id, nbr,
                    relation=self.graph[drop_id][nbr].get('relation', '')
                )
        self.graph.remove_node(drop_id)

        if self.last_landmark_id == drop_id:
            self.last_landmark_id = keep_id

    # ── Spatial scene consensus ────────────────────────────────────────────────

    def consolidate_scene_labels(self, cluster_radius=CLUSTER_RADIUS_M,
                                  single_instance_scenes=SINGLE_INSTANCE_SCENES):
        """
        Spatial fusion pass for scene labels.

        _fuse_observation's D1/D2/D3 voting only reconciles repeat
        observations of the SAME landmark over time. It has no way to notice
        that one landmark's confidently-wrong label disagrees with every
        landmark physically around it, or that some scene types should only
        ever claim one physical location in the level. Left alone, that lets
        a handful of confusing frames produce e.g. three different "imaging"
        landmarks scattered across the map when there's only one real
        imaging room.

        Two passes, run in order:

          1. SPATIAL CONSENSUS -- single-linkage cluster landmarks by
             physical proximity (anything within `cluster_radius` of another
             member joins the same cluster), pool every member's
             scene_label_votes together, and assign ONE winning label to the
             whole cluster. A single mislabeled landmark sitting among a
             dozen correctly-labeled neighbours gets outvoted.

          2. GLOBAL UNIQUENESS -- for labels in `single_instance_scenes`, if
             more than one cluster still claims the same label after pass 1,
             only the cluster with the larger total pooled vote mass keeps
             it; the rest fall back to their cluster's next-best label.

        Mutates `scene_label` on every node. Leaves `scene_label_votes`
        untouched as the audit trail of how each landmark got there. Safe to
        call any time -- e.g. periodically alongside `_merge_duplicates()`,
        or once after loading a saved graph into read-only mode before
        serving any llm.py-grounded navigation queries against it.
        """
        ids = list(self.graph.nodes)
        if not ids:
            return

        positions = np.array([self.graph.nodes[i]['position'] for i in ids])
        tree      = cKDTree(positions)

        # ── Single-linkage spatial clustering via union-find ────────────────
        parent = {nid: nid for nid in ids}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, nid in enumerate(ids):
            for j in tree.query_ball_point(positions[i], cluster_radius):
                union(nid, ids[j])

        clusters = {}
        for nid in ids:
            clusters.setdefault(find(nid), []).append(nid)

        # ── Pass 1: pool votes per cluster, assign one consensus label ──────
        cluster_winner = {}
        for root, members in clusters.items():
            pooled = {}
            for nid in members:
                for label, w in self.graph.nodes[nid].get('scene_label_votes', {}).items():
                    pooled[label] = pooled.get(label, 0.0) + w
            if not pooled:
                cluster_winner[root] = (None, 0.0)
                continue
            label = max(pooled, key=pooled.get)
            cluster_winner[root] = (label, pooled[label])
            for nid in members:
                self.graph.nodes[nid]['scene_label'] = label
            clusters[root] = (members, pooled)   # keep pooled votes for pass 2

        # ── Pass 2: enforce single-instance scenes ──────────────────────────
        if single_instance_scenes:
            claims = {}
            for root, (label, mass) in cluster_winner.items():
                if label in single_instance_scenes:
                    claims.setdefault(label, []).append((root, mass))

            for label, contenders in claims.items():
                if len(contenders) <= 1:
                    continue
                contenders.sort(key=lambda c: c[1], reverse=True)
                for root, _ in contenders[1:]:   # all but the strongest claim lose
                    members, pooled = clusters[root]
                    fallback  = {k: v for k, v in pooled.items() if k != label}
                    new_label = max(fallback, key=fallback.get) if fallback else None
                    for nid in members:
                        self.graph.nodes[nid]['scene_label'] = new_label
                    print(f"[Mapper] '{label}' claimed by a stronger cluster -- "
                          f"reassigned {len(members)} landmark(s) -> {new_label}")

        print(f"[Mapper] Scene consolidation: {len(clusters)} spatial clusters "
              f"from {len(ids)} landmarks")

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self):
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            nodes.append({
                "id":                nid,
                "position":          data.get("position"),
                "clip_embed":        data.get("clip_embed"),
                "clip_label":        data.get("clip_label"),
                "label_votes":       data.get("label_votes", {}),
                "scene_label":       data.get("scene_label"),
                "scene_label_votes": data.get("scene_label_votes", {}),
                "ocr_text":          data.get("ocr_text", ""),
                "timestamp":         data.get("timestamp"),
                "obs_count":         data.get("obs_count", 1),
                "saliency_mean":     data.get("saliency_mean"),
                "best_saliency":     data.get("best_saliency", 0.0),
            })

        edges = []
        for u, v, edata in self.graph.edges(data=True):
            edges.append({
                "source":   u,
                "target":   v,
                "relation": edata.get("relation", ""),
            })

        return {
            "version":            1,
            "spatial_threshold":  self.spatial_threshold,
            "semantic_threshold": self.semantic_threshold,
            "last_landmark_id":   self.last_landmark_id,
            "next_lm_id":         self._next_lm_id,
            "nodes":              nodes,
            "edges":              edges,
        }

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    def save(self, path):
        path     = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f, default=self._json_default)
        tmp_path.replace(path)
        print(f"[Mapper] Saved {len(self.graph.nodes)} landmarks "
              f"({len(self.graph.edges)} edges) → {path}")

    @classmethod
    def from_dict(cls, data):
        mapper = cls(
            spatial_threshold  = data.get("spatial_threshold",  SEMANTIC_THRESHOLD),
            semantic_threshold = data.get("semantic_threshold", SEMANTIC_THRESHOLD),
        )
        for node in data.get("nodes", []):
            nid   = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            mapper.graph.add_node(nid, **attrs)

        for edge in data.get("edges", []):
            mapper.graph.add_edge(
                edge["source"], edge["target"],
                relation=edge.get("relation", ""),
            )

        mapper.last_landmark_id = data.get("last_landmark_id")
        mapper._next_lm_id      = data.get(
            "next_lm_id",
            max(
                (int(n["id"].split("_")[1]) for n in data.get("nodes", [])),
                default=-1
            ) + 1,
        )
        mapper._rebuild_kdtree()
        print(f"[Mapper] Loaded {len(mapper.graph.nodes)} landmarks "
              f"({len(mapper.graph.edges)} edges) from disk")
        return mapper

    @classmethod
    def load(cls, path):
        with open(Path(path), "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_landmark(self, pos, clip_embedding, ocr_text, timestamp, distance,
                     saliency_mean=None, clip_label=None, clip_score=None,
                     scene_label=None, scene_score=None):
        """
        Main entry point — called once per frame when a salient object is detected.

        Parameters
        ----------
        pos           : array-like (3,)  — world position
        clip_embedding: np.ndarray (512,) or None — object-crop CLIP embedding
        ocr_text      : str or None
        timestamp     : float
        distance      : float — metric distance to object
        saliency_mean : float or None
        clip_label    : str or None — object-level class (from salient crop)
        clip_score    : float or None — confidence of clip_label
        scene_label   : str or None — room-level class (from full-frame classifier)
        """

        # 1. Normalise embedding
        if clip_embedding is not None:
            clip_embedding = np.array(clip_embedding, dtype=np.float32).flatten()
            norm           = np.linalg.norm(clip_embedding)
            clip_embedding = clip_embedding / norm if norm > 1e-6 else None

        # 2. Find best existing match
        best_match = self._find_best_match(pos, clip_embedding, clip_label, clip_score)

        # 3. Fuse or create
        if best_match is not None:
            self._fuse_observation(
                best_match, pos, clip_embedding, clip_label,
                clip_score, ocr_text, timestamp, distance,
                saliency_mean, scene_label, scene_score=scene_score,
            )
            landmark_id = best_match
        else:
            landmark_id      = f"lm_{self._next_lm_id}"
            self._next_lm_id += 1
            self.graph.add_node(landmark_id, **{
                'position'          : (pos.tolist()
                                       if isinstance(pos, np.ndarray) else pos),
                'clip_embed'        : (clip_embedding.tolist()
                                       if clip_embedding is not None
                                       else np.zeros(512).tolist()),
                'ocr_text'          : ocr_text or "",
                'timestamp'         : timestamp,
                'obs_count'         : 1,
                'clip_label'        : clip_label,
                'label_votes'       : {clip_label: 1} if clip_label else {},
                'scene_label'       : scene_label,
                'scene_label_votes' : (
                    # Seed with confidence-weighted initial vote + object prior
                    _initial_scene_votes(scene_label, scene_score, clip_label, clip_score)
                    if scene_label else {}
                ),
                'saliency_mean'     : saliency_mean,
                'best_saliency'     : saliency_mean or 0.0,
            })
            print(
                f"[Mapper] New landmark {landmark_id} at {pos} "
                f"| obj={clip_label} | scene={scene_label}"
            )

        # 4. Temporal edge
        if self.last_landmark_id and self.last_landmark_id != landmark_id:
            if not self.graph.has_edge(self.last_landmark_id, landmark_id):
                self.graph.add_edge(
                    self.last_landmark_id, landmark_id, relation="temporal"
                )

        self.last_landmark_id  = landmark_id
        self._obs_since_merge += 1

        # 5. Periodic duplicate merging + scene consensus (every 30 observations)
        if self._obs_since_merge >= 30:
            self._merge_duplicates()
            self.consolidate_scene_labels()
            self._obs_since_merge = 0

        self._rebuild_kdtree()
        return landmark_id

