from perception.saliency import BASNetSaliency
import perception.clipy as clipy

from navigation.mapper import SemanticMapper

SALIENCY_MEAN_THRESHOLD = 0.05

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

# model instances
saliency_detector = BASNetSaliency()
clip_model = clipy.CLIPModel()          # uses default "ViT-B/32"

# storage for landmarks
semantic_mapper = SemanticMapper(spatial_threshold=1.0, semantic_threshold=0.85)