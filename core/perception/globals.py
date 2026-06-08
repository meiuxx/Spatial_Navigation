# perception/globals.py
#
# Single source of truth for all model instances.
# One CLIP model loaded here serves both:
#   1. Object classification  (salient crop → TARGET_CLASSES label)
#   2. Scene classification   (full frame → room label via linear probe)

from perception.saliency import BASNetSaliency
import perception.clipy as clipy
from navigation.mapper import SemanticMapper

# ── Thresholds ────────────────────────────────────────────────────────────────

# BASNet: frames with mean saliency below this are considered featureless
# and skipped for landmark storage (occupancy map still updates every frame)
SALIENCY_MEAN_THRESHOLD = 0.043
# ── Object classes for CLIP ───────────────────────────────────────────────────
# Used when classifying the salient crop found by BASNet.
# These are open-vocabulary object descriptions — anything YOLO wasn't
# trained on can still be caught and labelled here.

TARGET_CLASSES = [

    # ── Navigation anchors (highest landmark value) ───────────────────────────
    "a hospital hallway",
    "a corridor intersection",
    "a room entrance doorway",
    "hospital double swing doors",
    "a support pillar",
    "a room divider",

    # ── Reception & waiting areas ─────────────────────────────────────────────
    "a hospital reception desk",
    "a waiting area chair",
    "a long couch or sofa",
    "a single armchair",

    # ── Clinical furniture ────────────────────────────────────────────────────
    "a hospital bed",
    "a medical examination table",
    "a wooden executive desk",
    "a medical office desk",
    "a small meeting table",
    "a doctor office chair",
    "a visitor sitting chair",
    "a glass display cabinet",
    "a document storage drawer",

    # ── Medical equipment ─────────────────────────────────────────────────────
    "a medical lightbox screen",
    "a medical monitor on a wheeled cart",
    "an IV pole",
    "a wheelchair",

    # ── Tech ──────────────────────────────────────────────────────────────────
    "a desktop screen",
    "a computer input device",

    # ── Utility & environment ─────────────────────────────────────────────────
    "a yellow cleaning cart",
    "a water dispenser",
    "a fluorescent panel",

    # ── Signage ───────────────────────────────────────────────────────────────
    "a wall mounted hospital sign",
    "a directional sign with arrows",
    "a health awareness poster on a wall",
    "a toilet sign",

    # ── Decor (useful for negative matching) ──────────────────────────────────
    "a decorative potted plant",
    "a decorative wall art",
]

# ── Model instances ───────────────────────────────────────────────────────────

# Saliency detector — finds the most visually salient region per frame
saliency_detector = BASNetSaliency()

# CLIP — loaded once, shared between object classification and scene classifier
clip_model = clipy.CLIPModel()
# ── Semantic graph ────────────────────────────────────────────────────────────

semantic_mapper = SemanticMapper(spatial_threshold=1.0, semantic_threshold=0.85)