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
OPT_TORSO_REF_DIR = "ffc_torso_ref_dir"
OPT_TORSO_STRENGTH = "ffc_torso_strength"
OPT_TORSO_DEPTH_WEIGHT = "ffc_torso_depth_weight"
OPT_TORSO_DEPTH_PREPROCESSOR = "ffc_torso_depth_preprocessor"
OPT_NAVEL_DETAILER = "ffc_navel_detailer"
OPT_NAVEL_GATE = "ffc_navel_gate"
OPT_NAVEL_TEMPLATE = "ffc_navel_template"
OPT_PROMPT_GATE = "ffc_prompt_gate"
OPT_BODY_GATE = "ffc_body_gate"
OPT_BODY_TOLERANCE = "ffc_body_tolerance"
OPT_REF_HEADS_TALL = "ffc_ref_heads_tall"
OPT_REF_SHOULDER_HIP = "ffc_ref_shoulder_hip"
OPT_REF_SHOULDER_HEADS = "ffc_ref_shoulder_heads"
OPT_REF_HIP_HEADS = "ffc_ref_hip_heads"
OPT_REF_LEG_FRACTION = "ffc_ref_leg_fraction"

BODY_GATE_MODES = ("off", "warn", "reject")
NAVEL_GATE_MODES = ("off", "warn")

DEFAULT_ENABLED = False
DEFAULT_STRENGTH = 0.85
DEFAULT_RESTORE = True
DEFAULT_VERIFY_THRESHOLD = 0.55
DEFAULT_LORA_NAME = ""
DEFAULT_LORA_WEIGHT = 0.0
DEFAULT_OUTFIT_STRENGTH = 0.0
DEFAULT_TORSO_REF_DIR = ""
DEFAULT_TORSO_STRENGTH = 0.45
DEFAULT_TORSO_DEPTH_WEIGHT = 0.0
DEFAULT_TORSO_DEPTH_PREPROCESSOR = "depth_midas"
DEFAULT_NAVEL_DETAILER = False
DEFAULT_NAVEL_GATE = "warn"
DEFAULT_NAVEL_TEMPLATE = ""
DEFAULT_PROMPT_GATE = True
DEFAULT_BODY_GATE = "warn"
DEFAULT_BODY_TOLERANCE = 0.15
# Reference body proportions for the body gate. "Heads" here are the gate's
# own units: OpenPose keypoint distances divided by the InsightFace face-bbox
# height -- NOT tape measurements. Keypoints sit at the joints (inside the
# silhouette), so these run systematically smaller than tape-measured widths
# (e.g. 1.53 vs 2.2 for shoulder width). Calibrated 2026-10-09 by running the
# gate's own measurement (measure_ratios over OpenPose keypoints + InsightFace
# head bbox) on the person's front full-body reference photo: the gate reads
# 1.43/1.53/1.07/8.46/0.52 on the real photo, and generated bodies that match
# the person read the same -- the old tape-based defaults warned on every
# generation. To recalibrate for anyone else: photograph them front
# full-body, run the same measurement, and set these five numbers to the
# result. Defaults are one specific person's gate-measured profile -- change
# them for anyone else.
DEFAULT_REF_HEADS_TALL = 8.46
DEFAULT_REF_SHOULDER_HIP = 1.43
DEFAULT_REF_SHOULDER_HEADS = 1.53
DEFAULT_REF_HIP_HEADS = 1.07
DEFAULT_REF_LEG_FRACTION = 0.52
