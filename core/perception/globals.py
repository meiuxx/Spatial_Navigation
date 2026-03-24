# globals.py
# This module initialises global instances used across the processing pipeline.
# It must be imported before using process_message.

from perception.saliency import BASNetSaliency
import perception.clipy as clipy
import perception.ocr as ocr

from navigation.mapper import SemanticMapper

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
# Global model instances (loaded once)
# ----------------------------------------------------------------------
saliency_detector = BASNetSaliency()
clip_model = clipy.CLIPModel()          # uses default "ViT-B/32"

# ----------------------------------------------------------------------
# Global storage for landmarks (now using SemanticMapper)
# ----------------------------------------------------------------------
semantic_mapper = SemanticMapper(spatial_threshold=1.0, semantic_threshold=0.85)