from perception.saliency import BASNetSaliency
import perception.clipy as clipy
from navigation.mapper import SemanticMapper

# ── Thresholds ────────────────────────────────────────────────────────────────

SALIENCY_MEAN_THRESHOLD = 0.043


TARGET_CLASSES = [

    # ── Navigation anchors ────────────────────────────────────────────────────
    "a hospital hallway",
    "a corridor intersection",
    "room entrance doorway",
    "hospital double swing doors",
    "a support pillar",
    "a green medical screen divider",

    # ── Reception & waiting areas ─────────────────────────────────────────────
    "a hospital reception desk",
    "a long couch or sofa",
    "a padded waiting area chair",
    "a wall mounted TV",
    "an emergency department sign in red font",

    # ── Clinical furniture ────────────────────────────────────────────────────
    "a hospital bed",
    "a medical examination table",
    "a surgical operating table",
    "a large office desk",
    "a meeting room table",
    "a wheeled office chair",
    "a meeting room chair",
    "a glass display cabinet",
    "a wooden wardrobe",
    "a metal locker cabinet",
    "a laboratory workbench",
    "a wall mounted cabinet",
    "a visitor chair",
    "a black metal display cabinet",

    # ── Medical equipment ─────────────────────────────────────────────────────
    "a medical monitor on a wheeled cart",
    "a blue medical oxygen cylinder",
    "an IV pole",
    "a wheelchair",
    "a patient monitor",
    "a surgical ceiling light",
    "a ceiling mounted display",
    "a anesthesia machine",
    "a ventilator machine",
    "an anesthesia/ventilator cart with vital signs monitor",
    "a ultrasound machine",
    "an MRI machine",
    "a surgical instrument tray",
    "a surgical equipment trolley",
    "a medical supply cart",
    "a medical cooler box",
    "a headwall panel",
    "a microscope",
    "a specimen tube rack",
    "a medical X-ray viewer lightbox",

    # ── Utility & environment ─────────────────────────────────────────────────
    "a yellow cleaning cart",
    "a water cooler",
    "a ceiling light panel",
    "a electrical outlet panel",
    "a fire alarm pull station",
    "a balloon bouquet",
    "a multi tier supply shelving unit",
    "a hospital gown hanging on rack",

    # ── Signage ───────────────────────────────────────────────────────────────
    "a directional sign with arrows",
    "a health awareness poster on a wall",
    "a toilet sign",
    "a yellow and blue laboratory warning sign",
    "a wall mounted bulletin board",

    # ── Bathroom ─────────────────────────────────────────────────────────────
    "a bathroom privacy cubicle",
    "a toilet cubicle door",
    "a wall mounted mirror",
    "a countertop washbasin",
    "a toilet",
    "a multi basin sink",
    "a soap dispenser",
    "a bathroom vanity light",
    "a bathroom mirror",

    # ── Cafeteria & dining ────────────────────────────────────────────────────
    "a high bar stool chair",
    "a cafeteria dining table with attached seating",
    "a vending machine with snacks and drinks",
    "a cold drinks dispenser",
    "a food service station counter",

    # ── Shop & pharmacy ───────────────────────────────────────────────────────
    "a retail shelf unit with price tags",
    "a pegboard display wall",
    "a chest freezer display unit",
    "a shop checkout counter",

    # ── Misc ──────────────────────────────────────────────────────────────────
    "a desktop monitor",
    "a decorative potted plant",
    "a decorative wall sign",

    # ── Laboratory ─────────────────────────────────────────────────────────────
    "a blue laboratory analysis machine",
    "a laboratory sample containers rack",
    "a laboratory conical flask",

]

# ── Model instances ───────────────────────────────────────────────────────────

# Saliency detector — finds the most visually salient region per frame
saliency_detector = BASNetSaliency()

# CLIP — loaded once, shared between object classification and scene classifier
clip_model = clipy.CLIPModel()
# ── Semantic graph ────────────────────────────────────────────────────────────

semantic_mapper = SemanticMapper(spatial_threshold=1.0, semantic_threshold=0.85)