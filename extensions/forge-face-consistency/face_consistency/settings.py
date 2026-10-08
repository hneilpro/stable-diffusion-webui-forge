"""Shared constants for Forge Face Consistency (no Forge imports)."""

SECTION_ID = "forge_face_consistency"
SECTION_LABEL = "Forge Face Consistency"

OPT_ENABLED = "ffc_enabled"
OPT_STRENGTH = "ffc_strength"
OPT_RESTORE = "ffc_restore"
OPT_VERIFY_THRESHOLD = "ffc_verify_threshold"
OPT_INSWAPPER_PATH = "ffc_inswapper_path"
OPT_LORA_NAME = "ffc_lora_name"
OPT_LORA_WEIGHT = "ffc_lora_weight"
OPT_OUTFIT_STRENGTH = "ffc_outfit_strength"
OPT_BODY_GATE = "ffc_body_gate"
OPT_BODY_TOLERANCE = "ffc_body_tolerance"
OPT_REF_HEADS_TALL = "ffc_ref_heads_tall"
OPT_REF_SHOULDER_HIP = "ffc_ref_shoulder_hip"
OPT_REF_SHOULDER_HEADS = "ffc_ref_shoulder_heads"
OPT_REF_HIP_HEADS = "ffc_ref_hip_heads"
OPT_REF_LEG_FRACTION = "ffc_ref_leg_fraction"

BODY_GATE_MODES = ("off", "warn", "reject")

DEFAULT_ENABLED = False
DEFAULT_STRENGTH = 0.85
DEFAULT_RESTORE = True
DEFAULT_VERIFY_THRESHOLD = 0.55
DEFAULT_LORA_NAME = ""
DEFAULT_LORA_WEIGHT = 0.0
DEFAULT_OUTFIT_STRENGTH = 0.0
DEFAULT_BODY_GATE = "warn"
DEFAULT_BODY_TOLERANCE = 0.15
# Reference body proportions ("heads" = head heights), measured from the
# person's front full-body reference photo. Defaults are one specific
# person's measurements -- change them for anyone else.
DEFAULT_REF_HEADS_TALL = 7.0
DEFAULT_REF_SHOULDER_HIP = 1.23
DEFAULT_REF_SHOULDER_HEADS = 2.2
DEFAULT_REF_HIP_HEADS = 1.8
DEFAULT_REF_LEG_FRACTION = 0.46
