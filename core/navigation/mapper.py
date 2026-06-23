# mapper.py
import json
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

# ── Tuning knobs (all in one place) ──────────────────────────────────────────

# KD-tree search radius.  Wide enough to absorb depth-estimation error at
# typical detection distances (up to ~8 m → ~1.5 m positional error).
SEARCH_RADIUS_M        = 3.0

# Fuse without any semantic check when the agent is very close and geometry
# alone is reliable.
CLOSE_RANGE_M          = 2.5
CLOSE_RANGE_SIM_FLOOR  = 0.50   # still require minimal semantic agreement

# Normal operating band: geometry + semantics must both agree.
SEMANTIC_THRESHOLD     = 0.72   # lowered from 0.85 — angle shift costs ~0.05–0.15

# When no close high-sim match exists, try a spatial-only fallback for very
# nearby candidates with at least a weak semantic signal.
SPATIAL_FALLBACK_DIST  = 1.5    # metres
SPATIAL_FALLBACK_SIM   = 0.55

# Merge duplicate nodes that drift together over time.
MERGE_SIM_THRESHOLD    = 0.88
MERGE_DIST_THRESHOLD   = 1.2    # metres

# Confidence-weighted embedding update — alpha scales with clip_score so
# blurry / low-confidence observations barely move the canonical embedding.
EMA_BASE_ALPHA         = 0.15
EMA_MAX_ALPHA          = 0.40

# Position update — weighted by inverse distance (closer = more accurate).
POS_ALPHA_BASE         = 0.25
POS_ALPHA_MAX          = 0.45


class SemanticMapper:
    """
    Maintains a spatial-semantic graph of observed landmarks.

    Key improvements over v1
    ------------------------
    * Wider KD-tree search radius to absorb depth-estimation error.
    * Distance-scaled semantic threshold: relax the gate when the geometry
      match is already very tight.
    * Confidence-weighted EMA: high clip_score observations steer the
      canonical embedding more than noisy low-confidence ones.
    * Label contradiction gate is advisory (scoring penalty) rather than
      hard reject — prevents one bad classification from permanently
      splitting a node.
    * Periodic duplicate-merge pass: collapses nodes that drifted together
      as the map refines.
    """

    def __init__(
        self,
        spatial_threshold : float = SEMANTIC_THRESHOLD,   # kept for API compat
        semantic_threshold: float = SEMANTIC_THRESHOLD,
    ):
        self.graph              = nx.Graph()
        self.spatial_threshold  = spatial_threshold
        self.semantic_threshold = semantic_threshold
        self.kd_tree            = None
        self.kd_tree_ids        = []
        self.last_landmark_id   = None
        self._obs_since_merge   = 0   # trigger periodic duplicate merging

        # FIX: landmark IDs must come from a monotonic counter, not
        # len(self.graph.nodes).  _merge_duplicates() removes nodes, so
        # graph size shrinks over time — basing new IDs on it eventually
        # produces an ID that already exists, and add_node() on an
        # existing ID silently overwrites it, fusing two unrelated
        # landmarks into one.  This also matters for save/load: the
        # counter must be restored from disk, not recomputed from the
        # loaded node count.
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
        """
        Return a composite match score for merging with an existing node.

        Score = semantic_similarity
              - label_mismatch_penalty   (soft penalty, not hard reject)
              + proximity_bonus          (tighter geometry → higher trust)

        Returns (score, sim) or (None, None) if the candidate should be
        skipped entirely (e.g. null embedding on both sides).
        """
        node     = self.graph.nodes[nid]
        node_pos = np.array(node['position'])
        dist_3d  = float(np.linalg.norm(np.array(pos) - node_pos))

        existing_embed = np.array(node['clip_embed'])
        e_norm         = np.linalg.norm(existing_embed)
        if e_norm < 1e-6:
            return None, None   # node has no usable embedding yet

        if clip_embedding is None:
            # No embedding available — fall back to geometry only when very close
            if dist_3d < SPATIAL_FALLBACK_DIST:
                return 0.60, 0.60   # synthetic mid-range score
            return None, None

        sim = float(np.dot(clip_embedding, existing_embed / e_norm))

        # Soft label-mismatch penalty instead of hard reject
        penalty = 0.0
        if clip_label and node.get('clip_label'):
            if clip_label != node['clip_label']:
                # Penalty scales with how confident the existing label is
                n_obs         = node.get('obs_count', 1)
                label_conf    = min(1.0, n_obs / 5.0)   # saturates at 5 observations
                penalty       = 0.08 * label_conf        # max –0.08

        # Proximity bonus: tight geometry buys some semantic slack
        proximity_bonus = max(0.0, (SPATIAL_FALLBACK_DIST - dist_3d) / SPATIAL_FALLBACK_DIST) * 0.06

        score = sim - penalty + proximity_bonus
        return score, sim

    # ── Match selection ───────────────────────────────────────────────────────

    def _find_best_match(self, pos, clip_embedding, clip_label, clip_score):
        """
        Return the best-matching existing node ID, or None.

        Three-tier fallback:
          1. Normal match  — within SEARCH_RADIUS_M, score > SEMANTIC_THRESHOLD
          2. Close match   — within CLOSE_RANGE_M, sim > CLOSE_RANGE_SIM_FLOOR
             (geometry is reliable at close range, relax semantics)
          3. Spatial-only  — within SPATIAL_FALLBACK_DIST, any positive sim
             (last resort; prevents obvious duplicates when CLIP is uncertain)
        """
        nearby_ids = self._find_nearby(pos, radius=SEARCH_RADIUS_M)
        if not nearby_ids:
            return None

        pos_arr    = np.array(pos)
        best_id    = None
        best_score = -np.inf

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

            # Tier 1: normal
            if score > self.semantic_threshold and score > best_score:
                best_score = score
                best_id    = nid

            # Tier 2: close-range
            if (dist_3d < CLOSE_RANGE_M
                    and sim  > CLOSE_RANGE_SIM_FLOOR
                    and sim  > close_sim):
                close_sim = sim
                close_id  = nid

            # Tier 3: spatial fallback
            if (dist_3d < SPATIAL_FALLBACK_DIST
                    and sim > 0.0
                    and sim > spatial_sim):
                spatial_sim = sim
                spatial_id  = nid

        if best_id is not None:
            return best_id
        if close_id is not None:
            return close_id
        if spatial_id is not None:
            return spatial_id
        return None

    # ── Observation fusion ────────────────────────────────────────────────────

    def _fuse_observation(self, nid, pos, clip_embedding, clip_label,
                          clip_score, ocr_text, timestamp, distance,
                          saliency_mean):
        """Update an existing node with a new observation."""
        node  = self.graph.nodes[nid]
        count = node.get('obs_count', 1)

        # A. Confidence-weighted position update
        #    Closer observations have lower noise; weight accordingly.
        dist_weight = 1.0 / (1.0 + max(distance, 0.1))
        pos_alpha   = np.clip(POS_ALPHA_BASE + dist_weight * 0.2,
                              POS_ALPHA_BASE, POS_ALPHA_MAX)
        old_pos     = np.array(node['position'])
        node['position'] = (
            (1 - pos_alpha) * old_pos + pos_alpha * np.array(pos)
        ).tolist()

        # B. Confidence-weighted EMA embedding update
        #    clip_score == 1.0 when unknown; use saliency as proxy then.
        if clip_embedding is not None:
            confidence = float(clip_score) if clip_score is not None else (
                min(1.0, saliency_mean * 10) if saliency_mean else 0.5
            )
            ema_alpha = np.clip(
                EMA_BASE_ALPHA + confidence * (EMA_MAX_ALPHA - EMA_BASE_ALPHA),
                EMA_BASE_ALPHA, EMA_MAX_ALPHA
            )
            old_embed        = np.array(node['clip_embed'])
            merged           = (1 - ema_alpha) * old_embed + ema_alpha * clip_embedding
            norm             = np.linalg.norm(merged)
            node['clip_embed'] = (merged / norm if norm > 1e-6 else merged).tolist()

        # C. Score-weighted label voting
        #    A high-confidence new observation gets more votes than a low one.
        if clip_label:
            vote_weight = max(1, round((clip_score or 0.5) * 3))
            votes       = node.setdefault(
                'label_votes',
                {node.get('clip_label', 'unknown'): count}
            )
            votes[clip_label] = votes.get(clip_label, 0) + vote_weight
            node['clip_label'] = max(votes, key=votes.get)

        # D. OCR: keep longest (most informative) string seen
        if ocr_text and len(ocr_text) > len(node.get('ocr_text', '')):
            node['ocr_text'] = ocr_text

        # E. Saliency: track best observation quality
        if saliency_mean is not None:
            node['best_saliency'] = max(
                node.get('best_saliency', 0.0), saliency_mean
            )

        node['obs_count'] = count + 1
        node['timestamp'] = timestamp

        print(
            f"[Mapper] Updated {nid} | obs={node['obs_count']} "
            f"dist={distance:.2f}m label={node['clip_label']} "
            f"votes={node.get('label_votes', {})}"
        )

    # ── Duplicate merging ─────────────────────────────────────────────────────

    def _merge_duplicates(self):
        """
        Scan all node pairs that are close in space *and* semantically similar
        and collapse the weaker one (fewer observations) into the stronger.

        Called every N observations to avoid O(n²) cost per frame.
        """
        ids   = list(self.graph.nodes)
        if len(ids) < 2:
            return

        positions = np.array([self.graph.nodes[i]['position'] for i in ids])
        tree      = cKDTree(positions)
        merged    = set()
        changed   = False

        for i, nid_a in enumerate(ids):
            if nid_a in merged:
                continue
            node_a   = self.graph.nodes[nid_a]
            embed_a  = np.array(node_a['clip_embed'])
            norm_a   = np.linalg.norm(embed_a)
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

                # Absorb weaker node into stronger
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
        """Fold drop_id's data into keep_id, then remove drop_id."""
        keep = self.graph.nodes[keep_id]
        drop = self.graph.nodes[drop_id]

        # Weighted position blend
        kc = keep.get('obs_count', 1)
        dc = drop.get('obs_count', 1)
        total = kc + dc
        keep['position'] = (
            (np.array(keep['position']) * kc + np.array(drop['position']) * dc) / total
        ).tolist()

        # Merge embeddings (obs-count weighted)
        merged = (
            np.array(keep['clip_embed']) * kc + np.array(drop['clip_embed']) * dc
        ) / total
        norm = np.linalg.norm(merged)
        keep['clip_embed'] = (merged / norm if norm > 1e-6 else merged).tolist()

        # Merge label votes
        for label, votes in drop.get('label_votes', {}).items():
            keep.setdefault('label_votes', {})[label] = (
                keep.get('label_votes', {}).get(label, 0) + votes
            )
        if keep.get('label_votes'):
            keep['clip_label'] = max(keep['label_votes'], key=keep['label_votes'].get)

        # Keep best OCR
        if len(drop.get('ocr_text', '')) > len(keep.get('ocr_text', '')):
            keep['ocr_text'] = drop['ocr_text']

        keep['obs_count']    = total
        keep['best_saliency'] = max(
            keep.get('best_saliency', 0.0), drop.get('best_saliency', 0.0)
        )

        # Rewire edges
        for nbr in list(self.graph.neighbors(drop_id)):
            if nbr != keep_id:
                self.graph.add_edge(keep_id, nbr,
                                    relation=self.graph[drop_id][nbr].get('relation', ''))
        self.graph.remove_node(drop_id)

        if self.last_landmark_id == drop_id:
            self.last_landmark_id = keep_id

    # ── Persistence ────────────────────────────────────────────────────────────

    def to_dict(self):
        """
        Serialise the full landmark graph to a plain-Python dict containing
        only JSON-safe types.  Every value already lives as a list/str/float
        on the node (positions and embeddings are `.tolist()`-ed at write
        time elsewhere in this file), so this is a straight pass-through —
        but we go field-by-field rather than dumping `dict(data)` so that
        any stray numpy scalar that sneaks onto a node in the future fails
        loudly in `_json_default` below instead of corrupting the file.
        """
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            nodes.append({
                "id":             nid,
                "position":       data.get("position"),
                "clip_embed":     data.get("clip_embed"),
                "clip_label":     data.get("clip_label"),
                "label_votes":    data.get("label_votes", {}),
                "ocr_text":       data.get("ocr_text", ""),
                "timestamp":      data.get("timestamp"),
                "obs_count":      data.get("obs_count", 1),
                "saliency_mean":  data.get("saliency_mean"),
                "best_saliency":  data.get("best_saliency", 0.0),
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
        """Fallback encoder for any numpy value that slipped through."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    def save(self, path):
        """
        Write the graph to disk as JSON.

        Writes to a temp file and renames over the target so a crash or
        kill mid-write (e.g. Ctrl+C during exploration) can never leave a
        truncated, unreadable graph file on disk — `load()` will always
        see either the previous good version or the new complete one.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f, default=self._json_default)
        tmp_path.replace(path)

        print(f"[Mapper] Saved {len(self.graph.nodes)} landmarks "
              f"({len(self.graph.edges)} edges) → {path}")

    @classmethod
    def from_dict(cls, data):
        """Rebuild a SemanticMapper from a dict produced by to_dict()."""
        mapper = cls(
            spatial_threshold=data.get("spatial_threshold", SEMANTIC_THRESHOLD),
            semantic_threshold=data.get("semantic_threshold", SEMANTIC_THRESHOLD),
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

        # FIX: restore the counter from disk rather than re-deriving it from
        # node count — node count can be lower than the highest ID ever
        # minted (nodes get removed by merging), so re-deriving it here
        # would reintroduce the exact collision bug the counter exists to
        # prevent, just one load cycle later.
        mapper._next_lm_id = data.get(
            "next_lm_id",
            max((int(n["id"].split("_")[1]) for n in data.get("nodes", [])), default=-1) + 1,
        )

        mapper._rebuild_kdtree()
        print(f"[Mapper] Loaded {len(mapper.graph.nodes)} landmarks "
              f"({len(mapper.graph.edges)} edges) from disk")
        return mapper

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_landmark(self, pos, clip_embedding, ocr_text, timestamp, distance,
                     saliency_mean=None, clip_label=None, clip_score=None):
        """
        Main entry point — called once per frame when a salient object is detected.
        """
        # 1. Normalise embedding
        if clip_embedding is not None:
            clip_embedding = np.array(clip_embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(clip_embedding)
            clip_embedding = clip_embedding / norm if norm > 1e-6 else None

        # 2. Find best existing match
        best_match = self._find_best_match(pos, clip_embedding, clip_label, clip_score)

        # 3. Fuse or create
        if best_match is not None:
            self._fuse_observation(
                best_match, pos, clip_embedding, clip_label,
                clip_score, ocr_text, timestamp, distance, saliency_mean
            )
            landmark_id = best_match
        else:
            landmark_id = f"lm_{self._next_lm_id}"
            self._next_lm_id += 1
            self.graph.add_node(landmark_id, **{
                'position'     : (pos.tolist() if isinstance(pos, np.ndarray) else pos),
                'clip_embed'   : (clip_embedding.tolist() if clip_embedding is not None
                                  else np.zeros(512).tolist()),
                'ocr_text'     : ocr_text or "",
                'timestamp'    : timestamp,
                'obs_count'    : 1,
                'clip_label'   : clip_label,
                'label_votes'  : {clip_label: 1} if clip_label else {},
                'saliency_mean': saliency_mean,
                'best_saliency': saliency_mean or 0.0,
            })
            print(f"[Mapper] New landmark {landmark_id} at {pos} | label={clip_label}")

        # 4. Temporal edge
        if self.last_landmark_id and self.last_landmark_id != landmark_id:
            if not self.graph.has_edge(self.last_landmark_id, landmark_id):
                self.graph.add_edge(self.last_landmark_id, landmark_id,
                                    relation="temporal")

        self.last_landmark_id   = landmark_id
        self._obs_since_merge  += 1

        # 5. Periodic duplicate merging (every 30 observations)
        if self._obs_since_merge >= 30:
            self._merge_duplicates()
            self._obs_since_merge = 0

        self._rebuild_kdtree()
        return landmark_id

    # ── Query (text → landmark) ─────────────────────────────────────────────────

    def query(self, text, clip_model, k=5, min_similarity=None, ocr_boost=0.1):
        """
        Rank landmarks by relevance to a natural-language query.

        Parameters
        ----------
        text : str
            The language query, e.g. "find the IV pole" or "go to the
            waiting area".
        clip_model : object with an `encode_text(list[str])` method
            Deliberately NOT imported at module level — mapper.py has no
            dependency on clipy.py or torch.  globals.py already imports
            both `SemanticMapper` and `clip_model`; if mapper.py imported
            clip_model itself you'd get a circular import (mapper → globals
            → mapper). The caller (process.py / whatever runs exploitation)
            already has clip_model in scope, so it's passed in instead.
            `encode_text` is expected to return something tensor-like
            supporting `[i]` then `.cpu().detach().numpy()` — i.e. CLIPModel
            from clipy.py works directly, with no adapter needed.
        k : int
            Max number of results to return.
        min_similarity : float or None
            Drop results below this cosine-similarity score. None = no
            filtering (caller decides via the returned score what counts
            as "found").
        ocr_boost : float
            Flat score bonus applied when `text` appears as a substring of
            a landmark's `ocr_text` (case-insensitive). CLIP's cross-modal
            similarity tends to sit lower than its within-modal similarity
            ("modality gap"), so an exact text-on-a-sign match is usually a
            much stronger signal than the embedding similarity alone would
            suggest — set to 0.0 to disable and rank on embedding
            similarity only. NOTE: this only helps if OCR text is actually
            attributable to the matched landmark — see the OCR/landmark
            attribution issue flagged separately; until that's fixed this
            boost can occasionally reward the wrong node.

        Returns
        -------
        list of dict, sorted by score descending, each:
            {"id", "score", "position", "clip_label", "ocr_text", "obs_count"}
        Empty list if the graph has no embedded landmarks yet, or if
        text encoding fails.
        """
        if len(self.graph.nodes) == 0:
            return []

        try:
            text_embed = clip_model.encode_text([text])[0].cpu().detach().numpy()
        except Exception as e:
            print(f"[Mapper] Query text-encoding failed: {e}")
            return []

        text_embed = np.asarray(text_embed, dtype=np.float32).flatten()
        t_norm = np.linalg.norm(text_embed)
        if t_norm < 1e-6:
            return []
        text_embed = text_embed / t_norm

        text_lower = text.lower().strip()
        results = []

        for nid, data in self.graph.nodes(data=True):
            embed = np.array(data.get('clip_embed', []), dtype=np.float32)
            norm  = np.linalg.norm(embed)
            if norm < 1e-6:
                continue  # never got a usable embedding — nothing to match against

            sim = float(np.dot(text_embed, embed / norm))

            ocr_text = data.get('ocr_text') or ""
            if ocr_boost and text_lower and text_lower in ocr_text.lower():
                sim += ocr_boost

            if min_similarity is not None and sim < min_similarity:
                continue

            results.append({
                "id":         nid,
                "score":      sim,
                "position":   data.get("position"),
                "clip_label": data.get("clip_label"),
                "ocr_text":   ocr_text,
                "obs_count":  data.get("obs_count", 1),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]
    