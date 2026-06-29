# perception/globals.py
#
# When GRAPH_READ_ONLY is True (set by main.py before this module is imported)
# the heavy models — BASNet, CLIP — are NOT loaded.  process.py returns at the
# top of its fast path before touching any of them, so the stubs below are
# never actually called.  This cuts startup time by ~10-30 s and saves ~1 GB
# of GPU memory during navigation-only runs.

import os

# ── Read-only flag ────────────────────────────────────────────────────────────
# main.py sets this to True before importing perception.globals when it detects
# a usable saved graph.  Check the env-var as a fallback so the flag survives
# even if the import order changes.
GRAPH_READ_ONLY: bool = os.environ.get("GRAPH_READ_ONLY", "0") == "1"

# ── Thresholds ────────────────────────────────────────────────────────────────
SALIENCY_MEAN_THRESHOLD = 0.043

# ── Scene classifier config ───────────────────────────────────────────────────
FINETUNED_MODEL_PATH = (
    "C:\\Users\\ALIENWARE\\Unity\\Spatial_Navigation_proj\\core\\"
    "perception\\clip_hospital_finetuned.pt"
)  # set to None to use zero-shot

SCENE_CLASSES = {
    "Administration":           "a hospital administration office with desks, computers, and office chairs",
    "bathroom":                 "a hospital public bathroom with toilet cubicles, wall mounted mirrors, and a multi basin sink",
    "conference room":          "a hospital meeting room with a large central table and chairs arranged around it",
    "corridor":                 "a hospital corridor with a black dado rail stripe on the walls and ceiling light panels overhead",
    "cafeteria":                "a hospital cafeteria with round wooden tables, bar stools, and red and blue vending machines along the wall",
    "emergency":                "a large open hospital area with an emergency department sign in red lettering on the wall",
    "examination":              "a hospital examination room with a white padded table on wheels, an IV pole, and a green folding privacy screen",
    "ultrasound":               "a small hospital imaging room with a white ultrasound console machine, hanging probe cables, and a large wall mounted monitor showing a scan",
    "MRI":                      "a hospital MRI machine with a donut-like shape",
    "lab":                      "a hospital laboratory with workbenches, a microscope, specimen tube racks, and blue analysis machines",
    "lobby":                    "a hospital main lobby with a large curved black reception counter in the centre, wheelchairs along the left wall, and waiting benches to the right",
    "main entrance":            "a wide open hospital entrance atrium with structural support pillars, ceiling light panels, and a large tiled floor with no reception desk visible",
    "OR_area":                  "a hospital operating theatre with a dark surgical table on a pedestal base and a large multi-bulb circular ceiling surgical light",
    "OR_sterilization":         "a hospital sterilization room with stainless steel sinks, scrub stations, and supply trolleys beside the entrance to an operating theatre",
    "patient room":             "a hospital inpatient room with a wheeled bed with a blue mattress and wooden headboard, a wooden wardrobe, and headwall outlet panels on the wall",
    "patient rooms hallway":    "a hospital ward corridor with wooden wardrobe cabinets and wheeled patient beds visible through open doorways on both sides",
    "pharamacy":                "a hospital pharmacy with pegboard display walls, retail shelving with price tags, and a blue Rx pharmacy sign",
    "reception-nurse station":  "a hospital nurse station with a raised desk counter, computers, and medical supply cabinets behind the counter",
    "supply room":              "a hospital supply room with multi-tier metal shelving units stacked with medical supplies and boxes",
    "VIP waiting area":         "a hospital waiting area with brown long couches and small brown one-seat couches around a centred coffee table",
    "standard waiting area":    "a hospital waiting area with rows of standard hospital seats",
    "administration lounge":    "a hospital cafeteria with rectangular wooden tables, metal chairs, and red and blue vending machines along the wall",
}

SCENE_CLASS_NAMES   = list(SCENE_CLASSES.keys())
SCENE_CLASS_PROMPTS = list(SCENE_CLASSES.values())

TARGET_CLASSES = [
    "a hospital hallway", "a corridor intersection", "room entrance doorway",
    "hospital double swing doors", "a support pillar", "a green medical screen divider",
    "a hospital reception desk", "a long couch or sofa", "a padded waiting area chair",
    "a wall mounted TV", "an emergency department sign in red font",
    "a hospital bed", "a medical examination table", "a surgical operating table",
    "a large office desk", "a meeting room table", "a wheeled office chair",
    "a meeting room chair", "a glass display cabinet", "a wooden wardrobe",
    "a metal locker cabinet", "a laboratory workbench", "a wall mounted cabinet",
    "a visitor chair", "a black metal display cabinet",
    "a medical monitor on a wheeled cart", "a blue medical oxygen cylinder",
    "an IV pole", "a wheelchair", "a patient monitor", "a surgical ceiling light",
    "a ceiling mounted display", "a anesthesia machine", "a ventilator machine",
    "an anesthesia/ventilator cart with vital signs monitor", "a ultrasound machine",
    "an MRI machine", "a surgical instrument tray", "a surgical equipment trolley",
    "a medical supply cart", "a medical cooler box", "a headwall panel",
    "a microscope", "a specimen tube rack", "a medical X-ray viewer lightbox",
    "a yellow cleaning cart", "a water cooler", "a ceiling light panel",
    "a electrical outlet panel", "a fire alarm pull station", "a balloon bouquet",
    "a multi tier supply shelving unit", "a hospital gown hanging on rack",
    "a directional sign with arrows", "a health awareness poster on a wall",
    "a toilet sign", "a yellow and blue laboratory warning sign",
    "a wall mounted bulletin board",
    "a bathroom privacy cubicle", "a toilet cubicle door", "a wall mounted mirror",
    "a countertop washbasin", "a toilet", "a multi basin sink", "a soap dispenser",
    "a bathroom vanity light", "a bathroom mirror",
    "a high bar stool chair", "a cafeteria dining table with attached seating",
    "a vending machine with snacks and drinks", "a cold drinks dispenser",
    "a food service station counter",
    "a retail shelf unit with price tags", "a pegboard display wall",
    "a chest freezer display unit", "a shop checkout counter",
    "a desktop monitor", "a decorative potted plant", "a decorative wall sign",
    "a blue laboratory analysis machine", "a laboratory sample containers rack",
    "a laboratory conical flask",
]

# ── Model instances ───────────────────────────────────────────────────────────

if GRAPH_READ_ONLY:
    # Navigation-only mode: skip loading heavy models entirely.
    # process.py returns before touching these, but define stubs so any
    # accidental import-time reference doesn't raise NameError.
    print("[Globals] Read-only mode — skipping BASNet, CLIP, and OCR model load.")
    saliency_detector = None
    clip_model        = None
else:
    from perception.saliency import BASNetSaliency
    import perception.clipy as clipy

    saliency_detector = BASNetSaliency()
    clip_model        = clipy.CLIPModel(finetuned_path=FINETUNED_MODEL_PATH)

# ── Semantic graph (always loaded) ────────────────────────────────────────────
from navigation.mapper import SemanticMapper
semantic_mapper = SemanticMapper(spatial_threshold=1.0, semantic_threshold=0.85)