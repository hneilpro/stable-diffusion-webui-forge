"""Forge Face Consistency — reference strength with max = face swap.

One control (Enable + reference image + Strength) keeps a chosen face
consistent across SDXL and Flux checkpoints:

- Strength = max (1.0): full InsightFace swap + restore in
  postprocess_image. Model-agnostic; works for SDXL and Flux.
- Strength < max on SDXL with a FaceID/InstantID ControlNet model present:
  inject a ControlNet unit at weight=strength (in-diffusion reference).
- Otherwise (Flux, or SDXL without an adapter): blended swap at
  blend=strength inside a feathered face mask.

A second, independent control — Reference outfit / object + Outfit
reference strength — injects a general IP-Adapter ControlNet unit
(Forge's built-in sd_forge_ipadapter) so a photo of an outfit, a prop,
or an environment steers the generation. Needs an IP-Adapter model
(e.g. ip-adapter_sdxl.safetensors) in models/ControlNet; the CLIP
vision encoder auto-downloads. SDXL and SD1.5 only — Flux has no
IP-Adapter in this Forge build and the outfit reference is skipped
with a logged reason.

Every run records the mode actually used, the strength, the outfit
reference (or its skip reason), the ArcFace similarity before/after,
and a head-height proportion assessment in the infotext. A downgrade
is always logged, never silent. Flux never receives SDXL adapters.

The body-proportion gate (Settings -> Forge Face Consistency) checks
the *body* the way the verify threshold checks the face: OpenPose
keypoints from the generated image are reduced to proportions
(heads-tall, shoulder:hip, widths in head units, leg fraction) and
compared against the reference profile. "warn" writes an infotext
warning; "reject" aborts the generation before the image is presented.
The pure measurement/assessment logic lives in
face_consistency.body_gate (importable and unit-tested without Forge).

The torso-consistency upgrade adds a third, independent control --
Torso reference strength -- driven by a reference *folder* configured
under Settings (the torso does not change per generation the way an
outfit does). It injects a general IP-Adapter unit conditioned on a
square torso crop (pose-based when the annotator cooperates, center
crop otherwise), plus an optional depth-lock ControlNet unit for
geometric anchoring. After generation, the navel gate
(face_consistency.torso, warn-only) checks the navel position against
the torso standard, and the optional navel detailer pass re-renders
the navel ROI at higher resolution -- the same crop/upscale/img2img/
paste-back pattern as the hand fix. The honest ceiling is stable
navel position plus clean rendering, never a pixel-identical navel.

The pure decision logic lives in face_consistency.logic (importable and
unit-tested without Forge); this module is the Forge-facing shell.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the sibling package importable both inside Forge and offline.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR = os.path.dirname(_THIS_DIR)
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

from face_consistency import logic  # noqa: E402
from face_consistency import settings as ffc_settings  # noqa: E402
from face_consistency import torso as torso_mod  # noqa: E402

try:
    import gradio as gr
    from modules import scripts, shared
    from modules.script_callbacks import on_ui_settings
    from modules.ui_components import InputAccordion
    _FORGE_AVAILABLE = True
except Exception:  # pragma: no cover - offline test environment
    gr = None  # type: ignore
    scripts = None  # type: ignore
    shared = None  # type: ignore
    on_ui_settings = None  # type: ignore
    InputAccordion = None  # type: ignore
    _FORGE_AVAILABLE = False

SCRIPT_NAME = "Forge Face Consistency"
_LOG_PREFIX = "[FaceConsistency]"


def _log(msg):
    print(f"{_LOG_PREFIX} {msg}")


def _opt(name, default):
    try:
        return shared.opts.data.get(name, default)
    except Exception:
        return default


def _family_inputs_forge(p):
    """(is_sdxl, class_name, filename_hint) feeding family detection."""
    sd_model = getattr(p, "sd_model", None)
    if sd_model is None and shared is not None:
        sd_model = getattr(shared, "sd_model", None)
    is_sdxl = bool(getattr(sd_model, "is_sdxl", False)) if sd_model is not None else False
    cls = type(sd_model).__name__ if sd_model is not None else ""
    hint = ""
    info = getattr(sd_model, "sd_checkpoint_info", None) if sd_model is not None else None
    if info is not None:
        hint = getattr(info, "filename", "") or ""
    if not hint and sd_model is not None:
        hint = getattr(sd_model, "filename", "") or ""
    return is_sdxl, cls, hint


def _detect_family_forge(p) -> str:
    is_sdxl, cls, hint = _family_inputs_forge(p)
    family = logic.detect_family(is_sdxl=is_sdxl, checkpoint_hint=hint, class_name=cls)
    _log(f"family detect: is_sdxl={is_sdxl} cls={cls!r} "
         f"hint={hint!r} -> {family}")
    return family


def _available_adapters():
    """(adapter_tuple_or_None, controlnet_names) via Forge ControlNet state."""
    try:
        from lib_controlnet import global_state

        names = list(global_state.controlnet_names or [])
        return logic.find_adapter_model(names), names
    except Exception as exc:
        _log(f"ControlNet state unavailable ({exc}); reference injection disabled")
        return None, []


def _decode_ref_string(value):
    """Reference image arriving as a string -> RGB numpy, or None.

    The WebUI passes a numpy array, but the API delivers image script
    args as base64 strings (optionally data-URI prefixed) — and a string
    is also how a same-machine caller passes a file path. Handle all
    three; never raise, since this runs inside someone else's
    generation (live failure 2026-10-07: np.asarray(str) is a 0-d array
    and shape[-1] died with "tuple index out of range").
    """
    text = value.decode("utf-8", "ignore") if isinstance(value, bytes) else value
    text = text.strip()
    if not text:
        return None
    if os.path.isfile(text):
        return _load_rgb_from_path(text)
    if text.startswith("data:"):
        text = text.split(",", 1)[-1] if "," in text else ""
    try:
        import base64
        import io

        from PIL import Image

        raw = base64.b64decode(text, validate=False)
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def _ref_to_rgb(ref_image):
    """UI/API reference (numpy / PIL / base64 / path) -> RGB numpy, or None."""
    if ref_image is None:
        return None
    if isinstance(ref_image, dict):  # gradio image-editor style payload
        return _ref_to_rgb(ref_image.get("image"))
    if isinstance(ref_image, (str, bytes)):
        arr = _decode_ref_string(ref_image)
        if arr is None:
            return None
    elif hasattr(ref_image, "convert"):  # PIL
        arr = np.asarray(ref_image.convert("RGB"))
    else:
        arr = np.asarray(ref_image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.size == 0:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _pp_to_rgb(pp):
    """Generated image (PIL or numpy) -> RGB numpy array, or None.

    Never raises. The live path hands us PIL; the offline tests hand
    us numpy -- handle both.
    """
    try:
        img = pp.image
        if hasattr(img, "convert"):
            try:
                img = img.convert("RGB")
            except Exception:
                pass
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.size == 0:
            return None
        return np.ascontiguousarray(arr[..., :3])
    except Exception:
        return None


def _ref_to_bgr(ref_image):
    """UI/API reference -> BGR numpy, or None."""
    rgb = _ref_to_rgb(ref_image)
    if rgb is None:
        return None
    return np.ascontiguousarray(rgb[..., ::-1])  # RGB -> BGR


def _load_rgb_from_path(path):
    """Decode an image file to an RGB numpy array, or None."""
    if not path:
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        pass
    try:
        import cv2

        bgr = cv2.imread(path)
        if bgr is None:
            return None
        return np.ascontiguousarray(bgr[..., ::-1])
    except Exception:
        return None


def _unit_engine(p, refs_dir):
    """FaceSwapEngine for reference selection, or None (never raises).

    Only folder references benefit from face-aware selection; the common
    single-image path keeps zero overhead.
    """
    if not refs_dir or not str(refs_dir).strip():
        return None
    try:
        return _get_engine(p)
    except Exception:
        return None


def representative_rgb_for_unit(ref_image, refs_dir, engine=None):
    """Concrete RGB image for a ControlNet unit.

    The unit conditions on one image, so a folder-only reference needs
    a representative file decoded here (the swap template path still
    uses the full folder). A non-empty folder overrides the single
    image, matching the UI wording. When ``engine`` is given, the
    sharpest *detectable* face wins instead of the first file in sorted
    order (a back-view or blurry photo must not become the identity
    conditioning). Returns (rgb_or_None, source_note).
    """
    folder = str(refs_dir).strip() if refs_dir else ""
    if folder:
        # N2: try every folder file in sorted order until one decodes;
        # a corrupt first file must not discard an otherwise usable folder.
        decoded = []
        skipped = []
        for rep_path in logic.list_ref_candidates(folder):
            rgb = _load_rgb_from_path(rep_path)
            if rgb is not None and rgb.size:
                decoded.append((rep_path, rgb))
            else:
                skipped.append(rep_path)
        if engine is not None and decoded:
            pick = _sharpest_decoded(decoded, engine.detect)
            if pick is not None:
                rep_path, rgb, note_extra = pick
                note = f"folder representative {rep_path} (sharpest face)"
                if note_extra:
                    note += f" ({note_extra})"
                return rgb, note
            # else: fall through to first-decodable below
        for rep_path, rgb in decoded:
            note = f"folder representative {rep_path}"
            if skipped:
                note += f" (skipped unreadable: {', '.join(skipped)})"
            return rgb, note
        if ref_image is not None:
            rgb = _ref_to_rgb(ref_image)
            if rgb is not None and rgb.size:
                return rgb, "single image (folder representative unreadable)"
        return None, "no readable image in reference folder"
    rgb = _ref_to_rgb(ref_image)
    if rgb is not None and rgb.size:
        return rgb, "single image"
    return None, "no reference image"


def _sharpest_decoded(decoded, detect):
    """(path, rgb, note_extra) with the sharpest detectable face, or None.

    ``detect`` is FaceSwapEngine.detect (bound). Never raises: any
    detection failure falls back to first-decodable selection in the
    caller.
    """
    try:
        from face_consistency.swap_engine import face_sharpness, largest

        best, best_s, noface = None, -1.0, []
        for rep_path, rgb in decoded:
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            try:
                faces = detect(bgr)
            except Exception:
                faces = None
            if not faces:
                noface.append(rep_path)
                continue
            s = face_sharpness(bgr, largest(faces))
            s = float(s) if s is not None else -1.0
            if s > best_s:
                best, best_s = (rep_path, rgb), s
        if best is None:
            return None
        extra = f"no face: {', '.join(noface)}" if noface else ""
        return best[0], best[1], extra
    except Exception:
        return None


def _get_engine(p):
    """FaceSwapEngine cached on ``p`` so a batch builds it once."""
    engine = getattr(p, "_ffc_engine", None)
    if engine is None:
        from face_consistency.swap_engine import FaceSwapEngine

        engine = FaceSwapEngine(
            inswapper_path=_opt(ffc_settings.OPT_INSWAPPER_PATH, "") or None)
        p._ffc_engine = engine
    return engine


def _verify_threshold():
    try:
        return float(_opt(ffc_settings.OPT_VERIFY_THRESHOLD,
                          ffc_settings.DEFAULT_VERIFY_THRESHOLD))
    except (TypeError, ValueError):
        return ffc_settings.DEFAULT_VERIFY_THRESHOLD


def _body_gate_config():
    """(mode, tolerance, reference) for the body-proportion gate."""
    mode = str(_opt(ffc_settings.OPT_BODY_GATE,
                    ffc_settings.DEFAULT_BODY_GATE) or "warn")
    try:
        tolerance = float(_opt(ffc_settings.OPT_BODY_TOLERANCE,
                               ffc_settings.DEFAULT_BODY_TOLERANCE))
    except (TypeError, ValueError):
        tolerance = ffc_settings.DEFAULT_BODY_TOLERANCE
    reference = {}
    for opt, default, key in (
            (ffc_settings.OPT_REF_HEADS_TALL,
             ffc_settings.DEFAULT_REF_HEADS_TALL, "heads_tall"),
            (ffc_settings.OPT_REF_SHOULDER_HIP,
             ffc_settings.DEFAULT_REF_SHOULDER_HIP, "shoulder_hip"),
            (ffc_settings.OPT_REF_SHOULDER_HEADS,
             ffc_settings.DEFAULT_REF_SHOULDER_HEADS, "shoulder_heads"),
            (ffc_settings.OPT_REF_HIP_HEADS,
             ffc_settings.DEFAULT_REF_HIP_HEADS, "hip_heads"),
            (ffc_settings.OPT_REF_LEG_FRACTION,
             ffc_settings.DEFAULT_REF_LEG_FRACTION, "leg_fraction")):
        try:
            reference[key] = float(_opt(opt, default))
        except (TypeError, ValueError):
            reference[key] = float(default)
    return mode, tolerance, reference


def _build_pose_detector():
    """Callable bgr -> list of raw keypoint lists, or None.

    Uses the ControlNet OpenPose annotator in-process when importable.
    A missing/unimportable annotator is not an error: the gate degrades
    to unmeasured with a loud infotext note.
    """
    try:
        from annotator.openpose import OpenposeDetector

        detector = OpenposeDetector()

        def run(image_bgr):
            import cv2

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            poses = detector.detect_poses(rgb) or []
            out = []
            for pose in poses:
                kps = getattr(getattr(pose, "body", None), "keypoints", None)
                if kps:
                    out.append(kps)
            return out

        return run
    except Exception as exc:
        _log(f"pose detector unavailable ({exc}); body gate unmeasured")
        return None


def _get_pose_detector(p):
    """OpenPose detector cached on ``p``, or None (never raises)."""
    det = getattr(p, "_ffc_pose_detector", "missing")
    if det == "missing":
        det = _build_pose_detector()
        p._ffc_pose_detector = det
    return det


def _detect_body_keypoints(p, image_bgr):
    """Normalized keypoints of the most complete pose, or None."""
    from face_consistency import body_gate as body_gate_mod

    run = _get_pose_detector(p)
    if run is None:
        return None
    try:
        poses = run(image_bgr) or []
    except Exception as exc:
        _log(f"pose detection failed ({exc})")
        return None
    best, best_n = None, -1
    for raw in poses:
        kp = body_gate_mod.normalize_keypoints(raw)
        if len(kp) > best_n:
            best, best_n = kp, len(kp)
    return best


def _head_bbox_for_gate(p, image_bgr):
    """(x0, y0, x1, y1) of the largest detected face, or None."""
    try:
        engine = _get_engine(p)
        faces = engine.detect(image_bgr)
        if not faces:
            return None
        from face_consistency.swap_engine import largest as _largest

        return tuple(float(v) for v in _largest(faces).bbox)
    except Exception as exc:
        _log(f"head bbox for body gate unavailable ({exc})")
        return None


def _run_body_gate(p, pp, precomputed=None):
    """Body-proportion gate on the generated image.

    Runs for every enabled generation whose mode is not "disabled"
    (the body comes from diffusion in every path; only the face is
    ever swapped). Warns in the infotext, or raises
    BodyProportionError in "reject" mode, when the measured
    proportions deviate from the reference profile beyond tolerance.
    Never raises in "warn"/"off" modes; a broken provider degrades to
    a loud infotext note, never a silent skip.

    ``precomputed`` is an optional (image_bgr, keypoints) tuple so the
    caller can share one pose detection across the body gate, the
    navel gate, and the navel detailer.
    """
    from face_consistency import body_gate as body_gate_mod

    mode, tolerance, reference = _body_gate_config()
    if str(mode).strip().lower() == "off":
        return
    if precomputed is not None:
        image_bgr, keypoints = precomputed
    else:
        image_bgr, keypoints = None, None
        try:
            import cv2

            image_bgr = cv2.cvtColor(np.asarray(pp.image), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            _log(f"body gate skipped: cannot decode image ({exc})")
            return
        keypoints = _detect_body_keypoints(p, image_bgr)
    if not keypoints:
        note = ("body gate: no pose detected"
                if _get_pose_detector(p) is not None
                else "body gate: pose detector unavailable")
        p.extra_generation_params["FaceConsistency body gate"] = note
        _log(f"WARNING: {note}")
        return
    bbox = _head_bbox_for_gate(p, image_bgr)
    head_h = (bbox[3] - bbox[1]) if bbox else None
    measured = body_gate_mod.measure_ratios(keypoints, head_h)
    verdict, details = body_gate_mod.assess(measured, reference, tolerance)
    if details:
        p.extra_generation_params["FaceConsistency body"] = \
            body_gate_mod.format_assessment(details)
    action = body_gate_mod.decide_action(verdict, mode)
    if action == "ok":
        if verdict == "unmeasured":
            note = "body gate: proportions unmeasurable (partial body?)"
            p.extra_generation_params["FaceConsistency body gate"] = note
            _log(f"WARNING: {note}")
        return
    failures = body_gate_mod.format_failures(details, tolerance)
    if action == "warn":
        p.extra_generation_params["FaceConsistency body gate"] = (
            f"WARNING: body proportions deviate from reference -- {failures}")
        _log(f"WARNING: body gate: {failures}")
        return
    # reject: fail loudly before the image is presented.
    msg = (f"body-proportion gate REJECTED this image: {failures} "
           f"[{body_gate_mod.format_assessment(details)}]")
    p.extra_generation_params["FaceConsistency body gate"] = msg
    _log(f"REJECT: {msg}")
    raise body_gate_mod.BodyProportionError(msg)


def _maybe_inject_character_lora(p, plan):
    """Append the configured character LoRA tag to the prompt.

    A character LoRA (trained on the person's photos) is the only
    inference-time mechanism that preserves *body type* as well as the
    face — face swap and FaceID/InstantID are face-only. The LoRA is
    configured once under Settings -> Forge Face Consistency and applied
    as ``<lora:name:weight>`` whenever the script is enabled. Never
    duplicates a tag the user already placed, and never raises.
    """
    try:
        name = str(_opt(ffc_settings.OPT_LORA_NAME,
                        ffc_settings.DEFAULT_LORA_NAME) or "").strip()
        try:
            weight = float(_opt(ffc_settings.OPT_LORA_WEIGHT,
                                ffc_settings.DEFAULT_LORA_WEIGHT) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if not name or weight <= 0:
            return
        marker = f"<lora:{name}:"
        tag = f"<lora:{name}:{weight:g}>"
        prompt = getattr(p, "prompt", "") or ""
        if marker in prompt:
            plan["lora_tag"] = f"{tag} (already present)"
            return
        p.prompt = (prompt + " " + tag).strip()
        allp = getattr(p, "all_prompts", None)
        if allp:
            p.all_prompts = [
                ap if marker in ap else (ap + " " + tag).strip() for ap in allp
            ]
        plan["lora_tag"] = tag
        p.extra_generation_params["FaceConsistency lora"] = tag
        _log(f"character LoRA injected: {tag}")
    except Exception as exc:
        _log(f"character LoRA injection skipped ({exc})")


def _maybe_inject_outfit_reference(p, plan, outfit_rgb, outfit_weight):
    """IP-Adapter outfit/object reference via Forge's built-in IP-Adapter.

    Independent of the face path: injects a second ControlNet unit using
    Forge's sd_forge_ipadapter (general image conditioning, not
    face-specific), so a photo of an outfit, a prop, or an environment
    steers the generation. A well-composed reference also transfers
    composition/proportions. Needs an IP-Adapter model
    (e.g. ip-adapter_sdxl.safetensors from h94/IP-Adapter) in
    models/ControlNet; the CLIP vision encoder auto-downloads on first
    use. SDXL and SD1.5 only — Flux has no IP-Adapter in this Forge
    build. Never raises; every outcome is recorded in the infotext.
    """
    try:
        family = plan.get("family") or _detect_family_forge(p)
        if family == "flux":
            note = ("outfit reference skipped: no Flux IP-Adapter "
                    "in this Forge build")
            p.extra_generation_params["FaceConsistency outfit ref"] = note
            _log(note)
            return
        _adapter, names = _available_adapters()
        model = logic.find_ipadapter_model(names)
        if model is None:
            face_only = logic.list_face_ipadapter_models(names)
            note = ("outfit reference skipped: no general IP-Adapter model "
                    "found; download ip-adapter_sdxl.safetensors "
                    "(h94/IP-Adapter sdxl_models) into models/ControlNet")
            if face_only:
                note += (" (face-only variants present, not used for "
                         f"outfits: {', '.join(face_only)})")
            p.extra_generation_params["FaceConsistency outfit ref"] = note
            _log(note)
            return
        preprocessor = logic.pick_ipadapter_preprocessor(model, family)
        unit = _build_controlnet_unit(preprocessor, model,
                                      float(outfit_weight), outfit_rgb)
        if _inject_controlnet_unit(p, unit):
            note = f"{model} @ {float(outfit_weight):.2f} ({preprocessor})"
            p.extra_generation_params["FaceConsistency outfit ref"] = note
            _log(f"outfit reference injected: {note}")
        else:
            note = "outfit reference skipped: no free ControlNet slot"
            p.extra_generation_params["FaceConsistency outfit ref"] = note
            _log(note)
    except Exception as exc:
        note = f"outfit reference failed: {exc}"
        try:
            p.extra_generation_params["FaceConsistency outfit ref"] = note
        except Exception:
            pass
        _log(note)


def _torso_representative_rgb(folder):
    """(path, rgb_or_None) -- first decodable image in the torso folder.

    Unlike the face path, no face-aware selection: the folder is
    expected to hold torso crops/photos, and sharpness-of-face is the
    wrong criterion. Never raises.
    """
    try:
        for rep_path in logic.list_ref_candidates(folder):
            rgb = _load_rgb_from_path(rep_path)
            if rgb is not None and rgb.size:
                return rep_path, rgb
    except Exception as exc:
        _log(f"torso folder listing failed ({exc})")
    return None, None


def _torso_square_crop(p, rgb):
    """Square torso crop for the IP-Adapter / depth units.

    Pose-based (shoulders->hips with margin) when the annotator finds a
    pose in the reference, otherwise a deterministic center square
    crop. Never raises; never returns None for a usable input.
    """
    try:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        kp = _detect_body_keypoints(p, bgr)
        if kp:
            box = torso_mod.torso_box_for_crop(kp, rgb.shape[1],
                                               rgb.shape[0])
            if box:
                x0, y0, x1, y1 = box
                _log(f"torso crop: pose-based box {box}")
                return np.ascontiguousarray(rgb[y0:y1, x0:x1])
    except Exception as exc:
        _log(f"torso crop: pose crop failed ({exc}); center crop")
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return np.ascontiguousarray(rgb[y0:y0 + side, x0:x0 + side])


def _maybe_inject_torso_reference(p, plan, torso_weight, depth_weight):
    """Torso consistency: IP-Adapter unit (+ optional depth-lock unit).

    A third, independent reference path beside the face and outfit
    slots: a general IP-Adapter ControlNet unit conditioned on a square
    torso crop from the configured reference folder, at
    ``torso_weight`` (default 0.45 -- below the weight where the
    adapter starts dominating the prompt). Optionally a depth ControlNet
    unit on the same crop for geometric anchoring (breast volume, waist
    curve). Needs the same general IP-Adapter model the outfit slot
    needs; Flux skips loudly. Never raises; every outcome is recorded
    in the infotext.
    """
    try:
        folder = str(_opt(ffc_settings.OPT_TORSO_REF_DIR,
                          ffc_settings.DEFAULT_TORSO_REF_DIR)
                     or "").strip()
        if not folder or not os.path.isdir(folder):
            note = ("torso reference skipped: ffc_torso_ref_dir not set "
                    "or not a folder (Settings -> Forge Face Consistency)")
            p.extra_generation_params["FaceConsistency torso ref"] = note
            _log(note)
            return
        family = plan.get("family") or _detect_family_forge(p)
        if family == "flux":
            note = ("torso reference skipped: no Flux IP-Adapter "
                    "in this Forge build")
            p.extra_generation_params["FaceConsistency torso ref"] = note
            _log(note)
            return
        rep_path, rgb = _torso_representative_rgb(folder)
        if rgb is None:
            note = (f"torso reference skipped: no readable image in "
                    f"{folder}")
            p.extra_generation_params["FaceConsistency torso ref"] = note
            _log(note)
            return
        crop = _torso_square_crop(p, rgb)
        _adapter, names = _available_adapters()
        injected = []
        if torso_weight > 0:
            model = logic.find_ipadapter_model(names)
            if model is None:
                note = ("torso IP-Adapter skipped: no general IP-Adapter "
                        "model found (same model the outfit slot needs: "
                        "ip-adapter_sdxl.safetensors in models/ControlNet)")
                _log(note)
            else:
                preprocessor = logic.pick_ipadapter_preprocessor(
                    model, family)
                unit = _build_controlnet_unit(preprocessor, model,
                                              float(torso_weight), crop)
                if _inject_controlnet_unit(p, unit):
                    injected.append(
                        f"ip-adapter {model} @ {float(torso_weight):.2f}")
                    _log(f"torso reference injected: {model} @ "
                         f"{float(torso_weight):.2f} from {rep_path}")
                else:
                    _log("torso IP-Adapter skipped: no free ControlNet slot")
        if depth_weight > 0:
            dmodel = logic.find_depth_model(names)
            if dmodel is None:
                _log("torso depth lock skipped: no depth ControlNet model "
                     "found in models/ControlNet")
            else:
                dpre = str(_opt(
                    ffc_settings.OPT_TORSO_DEPTH_PREPROCESSOR,
                    ffc_settings.DEFAULT_TORSO_DEPTH_PREPROCESSOR)
                    or "depth_midas").strip()
                unit = _build_controlnet_unit(dpre, dmodel,
                                              float(depth_weight), crop)
                if _inject_controlnet_unit(p, unit):
                    injected.append(
                        f"depth {dmodel} @ {float(depth_weight):.2f}")
                    _log(f"torso depth lock injected: {dmodel} @ "
                         f"{float(depth_weight):.2f} ({dpre})")
                else:
                    _log("torso depth lock skipped: no free ControlNet slot")
        p.extra_generation_params["FaceConsistency torso ref"] = (
            "injected: " + ", ".join(injected) if injected
            else "skipped (see log)")
    except Exception as exc:
        note = f"torso reference failed: {exc}"
        try:
            p.extra_generation_params["FaceConsistency torso ref"] = note
        except Exception:
            pass
        _log(note)


def _run_navel_gate(p, pp, image_rgb=None, image_bgr=None, keypoints=None):
    """Navel-position gate (warn-only) on the generated image.

    The expected navel pixel comes from the pose keypoints via the
    torso standard (midline, 15% waist->crotch); the detected navel
    comes from template-matching a reference navel crop inside the
    waist ROI. Deviations beyond the tolerance warn in the infotext.
    Never raises; a missing template or pose degrades to a loud note,
    never a silent skip. There is deliberately no "reject" mode: the
    honest ceiling is stable position, not pixel identity.
    """
    mode = str(_opt(ffc_settings.OPT_NAVEL_GATE,
                    ffc_settings.DEFAULT_NAVEL_GATE) or "off").strip().lower()
    if mode == "off":
        return
    try:
        if image_rgb is None:
            image_rgb = _pp_to_rgb(pp)
        if image_rgb is None:
            note = "navel gate: cannot decode generated image"
            p.extra_generation_params["FaceConsistency navel"] = note
            _log(f"WARNING: {note}")
            return
        if keypoints is None:
            try:
                if image_bgr is None:
                    import cv2

                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                keypoints = _detect_body_keypoints(p, image_bgr)
            except Exception as exc:
                note = f"navel gate: pose detection failed ({exc})"
                p.extra_generation_params["FaceConsistency navel"] = note
                _log(f"WARNING: {note}")
                return
        if not keypoints:
            note = "navel gate: no pose detected; navel position not verified"
            p.extra_generation_params["FaceConsistency navel"] = note
            _log(f"WARNING: {note}")
            return
        expected = torso_mod.expected_navel_px(keypoints)
        if expected is None:
            note = ("navel gate: shoulders/hips missing; "
                    "navel position not verified")
            p.extra_generation_params["FaceConsistency navel"] = note
            _log(f"WARNING: {note}")
            return
        template_path = str(_opt(ffc_settings.OPT_NAVEL_TEMPLATE,
                                 ffc_settings.DEFAULT_NAVEL_TEMPLATE)
                            or "").strip()
        if not template_path or not os.path.isfile(template_path):
            note = ("navel gate: no navel template configured "
                    "(ffc_navel_template); navel position not verified")
            p.extra_generation_params["FaceConsistency navel"] = note
            _log(f"WARNING: {note}")
            return
        tpl = _load_rgb_from_path(template_path)
        if tpl is None:
            note = ("navel gate: navel template unreadable; "
                    "navel position not verified")
            p.extra_generation_params["FaceConsistency navel"] = note
            _log(f"WARNING: {note}")
            return
        h, w = image_rgb.shape[:2]
        side = torso_mod.crop_side_px(w, h, 0.30, 96)
        box = torso_mod.square_box(w, h, expected[0], expected[1], side)
        if box is None:
            note = "navel gate: waist ROI unusable; not verified"
            p.extra_generation_params["FaceConsistency navel"] = note
            _log(f"WARNING: {note}")
            return
        x0, y0, x1, y1 = box
        roi = image_rgb[y0:y1, x0:x1]
        cx, cy, score = torso_mod.ncc_detect(roi, tpl)
        detected = (x0 + cx, y0 + cy) if cx is not None else None
        head_h = None
        if image_bgr is None:
            import cv2

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        bbox = _head_bbox_for_gate(p, image_bgr)
        if bbox:
            head_h = bbox[3] - bbox[1]
        dev_heads, warning = torso_mod.assess_navel(
            expected, detected, score, head_h)
        report = (f"expected=({expected[0]:.0f},{expected[1]:.0f}) "
                  f"detected="
                  f"({detected[0]:.0f},{detected[1]:.0f})"
                  if detected else "detected=n/a")
        report += f" score={score:.2f}"
        if dev_heads is not None:
            report += f" deviation={dev_heads:.2f} heads"
        if warning:
            p.extra_generation_params["FaceConsistency navel"] = (
                f"WARNING: {warning} [{report}]")
            _log(f"WARNING: navel gate: {warning} [{report}]")
        else:
            p.extra_generation_params["FaceConsistency navel"] = (
                f"ok [{report}]")
            _log(f"navel gate ok [{report}]")
    except Exception as exc:
        note = f"navel gate failed: {exc}"
        try:
            p.extra_generation_params["FaceConsistency navel"] = note
        except Exception:
            pass
        _log(note)


def _maybe_run_navel_detailer(p, pp, keypoints=None):
    """Navel detailer pass: crop the navel ROI, upscale, low-denoise
    img2img, feathered paste-back.

    The same pattern as the hand fix that works (crop -> 3x upscale ->
    img2img -> downscale -> feathered paste). Cleans up navel rendering
    and restores a stable navel position; it regenerates the navel,
    never transplants identity -- the honest ceiling. Default off
    (Settings -> Forge Face Consistency -> navel detailer). Never
    raises; the nested generation is flagged so this script ignores it
    (no recursion).
    """
    try:
        if not bool(_opt(ffc_settings.OPT_NAVEL_DETAILER,
                         ffc_settings.DEFAULT_NAVEL_DETAILER)):
            return
        if getattr(p, "_ffc_nested", False):
            return
        from PIL import Image

        image_rgb = _pp_to_rgb(pp)
        if image_rgb is None:
            note = "navel detailer skipped: cannot decode generated image"
            p.extra_generation_params[
                "FaceConsistency navel detailer"] = note
            _log(note)
            return
        if keypoints is None:
            import cv2

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            keypoints = _detect_body_keypoints(p, image_bgr)
        if not keypoints:
            note = "navel detailer skipped: no pose detected"
            p.extra_generation_params[
                "FaceConsistency navel detailer"] = note
            _log(note)
            return
        expected = torso_mod.expected_navel_px(keypoints)
        if expected is None:
            note = "navel detailer skipped: shoulders/hips missing"
            p.extra_generation_params[
                "FaceConsistency navel detailer"] = note
            _log(note)
            return
        h, w = image_rgb.shape[:2]
        side = torso_mod.crop_side_px(w, h, 0.18, 128)
        box = torso_mod.square_box(w, h, expected[0], expected[1], side)
        if box is None:
            note = "navel detailer skipped: navel ROI unusable"
            p.extra_generation_params[
                "FaceConsistency navel detailer"] = note
            _log(note)
            return
        x0, y0, x1, y1 = box
        crop = image_rgb[y0:y1, x0:x1]
        up = Image.fromarray(crop).resize(
            (crop.shape[1] * 3, crop.shape[0] * 3), Image.LANCZOS)
        fixed = _img2img_navel_pass(p, up)
        if fixed is None:
            note = ("navel detailer skipped: img2img pass failed "
                    "(see log)")
            p.extra_generation_params[
                "FaceConsistency navel detailer"] = note
            _log(note)
            return
        fixed_small = fixed.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        from face_consistency import blend as blend_mod

        mask = blend_mod.feather_mask((h, w), box, feather=6.0)
        out = blend_mod.blend_images(
            image_rgb, np.asarray(fixed_small.convert("RGB")), mask, 1.0)
        pp.image = Image.fromarray(out)
        note = (f"applied (ROI {x1 - x0}px at "
                f"({expected[0]:.0f},{expected[1]:.0f}), 3x, denoise 0.35)")
        p.extra_generation_params["FaceConsistency navel detailer"] = note
        _log(f"navel detailer {note}")
    except Exception as exc:
        note = f"navel detailer failed: {exc}"
        try:
            p.extra_generation_params[
                "FaceConsistency navel detailer"] = note
        except Exception:
            pass
        _log(note)


def _img2img_navel_pass(p, init_pil):
    """One low-denoise img2img pass over the upscaled navel crop.

    Returns a PIL RGB image, or None on any failure. The processing
    object is flagged _ffc_nested so this script's own hooks ignore the
    nested generation (no recursion, no double gates).
    """
    try:
        from modules import processing, shared

        w, h = init_pil.size
        user_neg = getattr(p, "negative_prompt", "") or ""
        neg = ("deformed, disfigured, cartoon, drawing, blurry, "
               "watermark, text")
        p2 = processing.StableDiffusionProcessingImg2Img(
            init_images=[init_pil],
            prompt=("close-up of a natural human navel, smooth realistic "
                    "skin, photorealistic, high detail"),
            negative_prompt=(user_neg + ", " + neg).strip(", "),
            seed=getattr(p, "seed", -1),
            steps=20,
            cfg_scale=float(getattr(p, "cfg_scale", 7.0) or 7.0),
            denoising_strength=0.35,
            width=w,
            height=h,
            sd_model=getattr(shared, "sd_model", None),
        )
        p2.script_args = []
        p2._ffc_nested = True
        processed = processing.process_images(p2)
        images = getattr(processed, "images", None) or []
        if not images:
            _log("navel detailer: img2img returned no images")
            return None
        return images[0].convert("RGB")
    except Exception as exc:
        _log(f"navel detailer: img2img pass failed ({exc})")
        return None


def _build_controlnet_unit(preprocessor_name, model_name, weight, ref_rgb):
    """ControlNetUnit matching this clone's lib_controlnet.external_code.

    The unit image is the raw RGB numpy array, exactly as the ControlNet
    UI supplies it (see ControlNet.get_input_data / HWC3 handling).
    """
    from lib_controlnet.external_code import ControlNetUnit

    return ControlNetUnit(
        enabled=True,
        module=preprocessor_name,
        model=model_name,
        weight=float(weight),
        image=np.asarray(ref_rgb),
        resize_mode="Crop and Resize",
        guidance_start=0.0,
        guidance_end=1.0,
    )


def _inject_controlnet_unit(p, unit):
    """Place ``unit`` in the first free ControlNet slot of p.script_args.

    Returns True on success. ControlNet reads its units from its own slice
    of the flat script_args list (modules/scripts.py ScriptRunner), so we
    rewrite that slice in place. API-supplied dict units are converted by
    ControlNet itself; we always write a ControlNetUnit object.
    """
    runner = getattr(p, "scripts", None)
    if runner is None:
        return False
    raw_args = getattr(p, "script_args", None)
    if not raw_args:
        return False
    # Forge hands p.script_args over as an immutable tuple (live round 3,
    # 2026-10-08: "tuple object does not support item assignment").
    # Rewrite it as a list — ScriptRunner slices it either way.
    script_args = list(raw_args)
    for script in getattr(runner, "alwayson_scripts", []) or []:
        title = ""
        try:
            title = (script.title() or "").lower()
        except Exception:
            pass
        if "controlnet" not in title:
            continue
        start, end = script.args_from, script.args_to
        if start is None or end is None or end <= start:
            return False
        for idx in range(start, end):
            current = script_args[idx]
            enabled = True
            if isinstance(current, dict):
                enabled = bool(current.get("enabled", True))
            else:
                enabled = bool(getattr(current, "enabled", True))
            if not enabled:
                script_args[idx] = unit
                p.script_args = script_args
                return True
        return False  # all slots busy
    return False


if _FORGE_AVAILABLE:

    class FaceConsistencyScript(scripts.Script):
        sorting_priority = 11  # just after ControlNet (10)

        def title(self):
            return SCRIPT_NAME

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            default_enabled = bool(_opt(ffc_settings.OPT_ENABLED, ffc_settings.DEFAULT_ENABLED))
            default_strength = float(_opt(ffc_settings.OPT_STRENGTH, ffc_settings.DEFAULT_STRENGTH))
            default_restore = bool(_opt(ffc_settings.OPT_RESTORE, ffc_settings.DEFAULT_RESTORE))
            default_outfit_strength = float(_opt(ffc_settings.OPT_OUTFIT_STRENGTH,
                                                 ffc_settings.DEFAULT_OUTFIT_STRENGTH))
            default_torso_strength = float(_opt(ffc_settings.OPT_TORSO_STRENGTH,
                                                ffc_settings.DEFAULT_TORSO_STRENGTH))
            with InputAccordion(default_enabled, label=SCRIPT_NAME,
                                elem_id="ffc_accordion") as enabled:
                ref_image = gr.Image(label="Reference face", type="numpy")
                refs_dir = gr.Textbox(
                    label="Reference images folder (optional)",
                    placeholder="Folder of reference photos; overrides the single image above",
                )
                strength = gr.Slider(
                    label="Reference strength (max = full face swap)",
                    minimum=0.0, maximum=1.0, step=0.01, value=default_strength)
                restore = gr.Checkbox(label="Restore face detail after swap", value=default_restore)
                outfit_image = gr.Image(
                    label="Reference outfit / object (optional)",
                    type="numpy")
                outfit_strength = gr.Slider(
                    label="Outfit reference strength (0 = off)",
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=default_outfit_strength)
                torso_strength = gr.Slider(
                    label="Torso reference strength (0 = off)",
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=default_torso_strength)
            self.infotext_fields = [
                (enabled, "FaceConsistency enabled"),
                (strength, "FaceConsistency strength"),
            ]
            return (enabled, ref_image, refs_dir, strength, restore,
                    outfit_image, outfit_strength, torso_strength)

        # -- decision + ControlNet injection (runs before sampling) -------
        # New args (outfit_image, outfit_strength, torso_strength) append
        # at the end with defaults, so older 5-arg and 7-arg API calls
        # keep working.
        def before_process(self, p, enabled=False, ref_image=None, refs_dir="",
                           strength=1.0, restore=True,
                           outfit_image=None, outfit_strength=0.0,
                           torso_strength=0.0):
            # New generation: drop any engine cached by a previous run so
            # a changed inswapper path is honoured.
            if hasattr(p, "_ffc_engine"):
                del p._ffc_engine
            plan = {"enabled": bool(enabled), "mode": "disabled",
                    "strength": logic.clamp_strength(strength), "restore": bool(restore),
                    "downgraded": False, "reason": "disabled",
                    "verify_threshold": _verify_threshold()}
            p._ffc_plan = plan
            if not enabled:
                return
            has_face_ref = (ref_image is not None
                            or (refs_dir and str(refs_dir).strip()))
            outfit_rgb = _ref_to_rgb(outfit_image)
            outfit_w = logic.clamp_strength(outfit_strength)
            torso_w = logic.clamp_strength(torso_strength)
            torso_dir = str(_opt(ffc_settings.OPT_TORSO_REF_DIR,
                                 ffc_settings.DEFAULT_TORSO_REF_DIR)
                            or "").strip()
            try:
                depth_w = float(_opt(ffc_settings.OPT_TORSO_DEPTH_WEIGHT,
                                     ffc_settings.DEFAULT_TORSO_DEPTH_WEIGHT)
                                or 0.0)
            except (TypeError, ValueError):
                depth_w = 0.0
            has_torso_ref = bool(torso_dir) and (torso_w > 0 or depth_w > 0)
            if not has_face_ref and outfit_rgb is None and not has_torso_ref:
                plan["reason"] = ("enabled but no face, outfit, or torso "
                                   "reference given")
                _log("enabled with no reference; doing nothing")
                return
            _maybe_inject_character_lora(p, plan)
            try:
                family = _detect_family_forge(p)
                plan["family"] = family
                # Detection diagnostics: if the family ever looks wrong,
                # this line says exactly why (live 2026-10-08: "other" on
                # an SDXL checkpoint).
                is_sdxl, cls_name, hint = _family_inputs_forge(p)
                p.extra_generation_params["FaceConsistency family detail"] = (
                    f"is_sdxl={is_sdxl} cls={cls_name!r} hint={hint!r}")
                if has_face_ref:
                    adapter, _names = _available_adapters()
                    mode, downgraded, reason = logic.decide_mode(
                        plan["strength"], family, adapter is not None)
                    plan.update(downgraded=downgraded, reason=reason)
                    if mode == "controlnet" and adapter is not None:
                        kind, preprocessor_name, model_name = adapter
                        # F1: the unit needs a concrete image. A folder-only (or
                        # folder-overriding) reference must hand ControlNet a
                        # decoded representative, never np.asarray(None).
                        # With an engine, the sharpest detectable face wins over
                        # the first file in sorted order.
                        unit_rgb, unit_src = representative_rgb_for_unit(
                            ref_image, refs_dir, engine=_unit_engine(p, refs_dir))
                        if unit_rgb is None:
                            mode = "blended-swap"
                            plan.update(mode=mode, downgraded=True,
                                        reason=f"no reference image for ControlNet unit "
                                               f"({unit_src}); using blended swap")
                            _log(f"ControlNet unit image unavailable ({unit_src}); "
                                 f"downgrading to blended swap")
                        else:
                            unit = _build_controlnet_unit(preprocessor_name, model_name,
                                                          plan["strength"], unit_rgb)
                            plan["controlnet_image_source"] = unit_src
                            if _inject_controlnet_unit(p, unit):
                                plan["mode"] = kind
                                plan["controlnet_model"] = model_name
                                _log(f"injected ControlNet {kind} unit "
                                     f"(model={model_name}, weight={plan['strength']:.2f}, "
                                     f"image={unit_src})")
                            else:
                                mode = "blended-swap"
                                plan.update(mode=mode, downgraded=True,
                                            reason="ControlNet injection failed (no free slot); "
                                                   "using blended swap")
                                _log("ControlNet injection failed; downgrading to blended swap")
                    elif mode == "controlnet":
                        mode = "blended-swap"
                        plan.update(mode=mode, downgraded=True,
                                    reason="adapter reported available but unresolved; "
                                           "using blended swap")
                    else:
                        plan["mode"] = mode
                if has_face_ref and plan["mode"] in ("swap", "blended-swap"):
                    plan["ref_bgr"] = _ref_to_bgr(ref_image)
                    plan["refs_dir"] = str(refs_dir).strip() if refs_dir else ""
                if not has_face_ref:
                    # Faceless run: the face decision above never ran.
                    if outfit_rgb is not None and has_torso_ref:
                        plan["mode"] = "outfit+torso"
                        plan["reason"] = ("no face reference; outfit and "
                                           "torso references only")
                    elif has_torso_ref:
                        plan["mode"] = "torso-only"
                        plan["reason"] = "no face reference; torso reference only"
                    else:
                        plan["mode"] = "outfit-only"
                        plan["reason"] = ("no face reference; outfit "
                                           "reference only")
                if outfit_rgb is not None and outfit_w > 0:
                    _maybe_inject_outfit_reference(p, plan, outfit_rgb,
                                                   outfit_w)
                if has_torso_ref:
                    _maybe_inject_torso_reference(p, plan, torso_w, depth_w)
            except Exception as exc:
                # Setup failure must never kill the host generation or
                # vanish silently: live round 2 (2026-10-08) showed an
                # unguarded raise in this block leaves plan mode at
                # "disabled" with nothing in the infotext to explain it.
                # Record the failure, fall back to a blended swap, and
                # let generation continue.
                import traceback

                traceback.print_exc()
                plan["error"] = str(exc)
                if has_face_ref:
                    plan["ref_bgr"] = _ref_to_bgr(ref_image)
                    plan["refs_dir"] = str(refs_dir).strip() if refs_dir else ""
                    plan.update(mode="blended-swap", downgraded=True,
                                reason=f"setup failed ({exc}); using blended swap")
                else:
                    plan.update(mode="outfit-only", downgraded=True,
                                reason=f"setup failed ({exc}); "
                                       f"outfit reference only")
                _log(f"setup FAILED ({exc}); falling back")
            p.extra_generation_params.update({
                "FaceConsistency mode": plan["mode"],
                "FaceConsistency strength": f"{plan['strength']:.2f}",
                "FaceConsistency family": plan.get("family", "unknown"),
            })
            if plan.get("error"):
                p.extra_generation_params["FaceConsistency error"] = plan["error"]
            if plan["downgraded"]:
                p.extra_generation_params["FaceConsistency downgrade"] = plan["reason"]
                _log(f"downgrade: {plan['reason']}")
            _log(f"mode={plan['mode']} strength={plan['strength']:.2f} "
                 f"family={plan.get('family', 'unknown')} ({plan['reason']})")

        # -- swap / blended swap (per generated image) --------------------
        def postprocess_image(self, p, pp, enabled=False, ref_image=None,
                              refs_dir="", strength=1.0, restore=True,
                              outfit_image=None, outfit_strength=0.0,
                              torso_strength=0.0):
            plan = getattr(p, "_ffc_plan", None)
            if not plan or not plan.get("enabled"):
                return
            if getattr(p, "_ffc_nested", False):
                return  # nested navel-detailer generation: leave it alone
            # One pose detection per image, shared by the body gate, the
            # navel gate and the navel detailer. The body comes from
            # diffusion in every mode (only the face is ever swapped),
            # so the gates run before the mode early-return; in "reject"
            # mode a failing body aborts here, before the image is
            # presented. The face swap never moves the torso, so the
            # keypoints stay valid for the detailer after the swap.
            image_rgb, image_bgr, keypoints = None, None, None
            if plan["mode"] != "disabled":
                image_rgb = _pp_to_rgb(pp)
                if image_rgb is None:
                    _log("gates skipped: cannot decode generated image")
                else:
                    try:
                        import cv2

                        image_bgr = cv2.cvtColor(image_rgb,
                                                 cv2.COLOR_RGB2BGR)
                        keypoints = _detect_body_keypoints(p, image_bgr)
                    except Exception as exc:
                        _log(f"pose detection for gates failed ({exc})")
                _run_body_gate(p, pp, precomputed=(image_bgr, keypoints))
                _run_navel_gate(p, pp, image_rgb=image_rgb,
                                image_bgr=image_bgr, keypoints=keypoints)
            if plan["mode"] not in ("swap", "blended-swap"):
                # No swap ran: the ControlNet reference path forms
                # identity during diffusion, and faceless runs have no
                # face reference at all. Record n/a explicitly instead of
                # staying silent (F7).
                why = ("no face reference"
                       if plan["mode"] in ("outfit-only", "torso-only",
                                           "outfit+torso")
                       else "controlnet")
                p.extra_generation_params.setdefault(
                    "FaceConsistency similarity before", f"n/a ({why})")
                p.extra_generation_params.setdefault(
                    "FaceConsistency similarity after", f"n/a ({why})")
                if plan["mode"] != "disabled":
                    _maybe_run_navel_detailer(p, pp, keypoints=keypoints)
                return
            try:
                from face_consistency import blend as blend_mod
                import cv2

                ref_source = plan.get("refs_dir") or plan.get("ref_bgr")
                if ref_source is None:
                    ref_source = _ref_to_bgr(ref_image)
                # F5: engine (FaceAnalysis + swapper + template) is cached
                # on p and reused for every image in this generation.
                engine = _get_engine(p)
                pil_image = pp.image  # PIL, RGB
                target_bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
                swapped_bgr, info = engine.swap(
                    target_bgr, ref_source, restore=plan["restore"])
                if plan["mode"] == "blended-swap":
                    faces = engine.app.get(swapped_bgr)
                    if faces:
                        from face_consistency.swap_engine import largest as _largest

                        face = _largest(faces)
                        mask = blend_mod.feather_mask(
                            swapped_bgr.shape[:2], face.bbox, feather=8.0)
                        swapped_bgr = blend_mod.blend_images(
                            target_bgr, swapped_bgr, mask, plan["strength"])
                    else:
                        # F8: never silently return a full-strength swap
                        # when the user asked for a partial blend.
                        plan["blend_skipped"] = (
                            "no face detected in swapped image; "
                            "returned full-strength swap")
                        p.extra_generation_params[
                            "FaceConsistency blend skipped"] = plan["blend_skipped"]
                        _log(f"blended swap skipped: {plan['blend_skipped']}")
                out_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
                from PIL import Image

                pp.image = Image.fromarray(out_rgb)
                plan["last_info"] = info
                p.extra_generation_params.update({
                    "FaceConsistency similarity before": (
                        f"{info['similarity_before']:.3f}"
                        if info.get("similarity_before") is not None else "n/a"),
                    "FaceConsistency similarity after": (
                        f"{info['similarity_after']:.3f}"
                        if info.get("similarity_after") is not None else "n/a"),
                    "FaceConsistency restored by": info.get("restored_by", "none"),
                })
                ratio = info.get("face_size_ratio")
                if ratio is not None:
                    p.extra_generation_params["FaceConsistency face size"] = (
                        f"{ratio:.1%} of image")
                    if ratio < 0.02:
                        p.extra_generation_params["FaceConsistency note"] = (
                            "small target face (<2% of image); a closer crop "
                            "swaps more reliably")
                # Proportion gate: head height vs frame height, measured
                # from the swapped face bbox. Implausibly small heads get
                # an honest warning, never a silent pass.
                bbox = info.get("face_bbox")
                if bbox:
                    try:
                        head_ratio, head_warn = logic.head_height_assessment(
                            float(bbox[3]) - float(bbox[1]),
                            float(swapped_bgr.shape[0]))
                        if head_ratio > 0:
                            p.extra_generation_params[
                                "FaceConsistency head height"] = (
                                f"{head_ratio:.1%} of frame")
                        if head_warn:
                            p.extra_generation_params[
                                "FaceConsistency proportion"] = head_warn
                            _log(f"WARNING: {head_warn}")
                    except Exception:
                        pass
                # F4: the verify threshold is consumed here — a swap that
                # lands below it is warned on, not reported like a pass.
                threshold = plan.get("verify_threshold", _verify_threshold())
                sim_after = info.get("similarity_after")
                if logic.is_below_threshold(sim_after, threshold):
                    plan["verify_below_threshold"] = True
                    p.extra_generation_params["FaceConsistency verify"] = (
                        f"below threshold ({sim_after:.3f} < {threshold:.2f})")
                    _log(f"WARNING: {plan['mode']} similarity {sim_after:.3f} "
                         f"below verify threshold {threshold:.2f}")
                _log(f"{plan['mode']} done: similarity "
                     f"{info.get('similarity_before')} -> {info.get('similarity_after')} "
                     f"(restored by {info.get('restored_by')})")
            except Exception as exc:
                p.extra_generation_params["FaceConsistency error"] = str(exc)
                _log(f"{plan['mode']} FAILED: {exc}")
            # Navel detailer runs last, on the final pixels (after any
            # face swap), in every non-disabled mode.
            _maybe_run_navel_detailer(p, pp, keypoints=keypoints)

        def postprocess(self, p, processed, *args):
            plan = getattr(p, "_ffc_plan", None)
            if not plan or not plan.get("enabled"):
                return
            note = (f"FaceConsistency: mode={plan['mode']}, "
                    f"strength={plan['strength']:.2f}")
            if plan.get("downgraded"):
                note += f" (downgraded: {plan['reason']})"
            if plan.get("blend_skipped"):
                note += f" (blend skipped: {plan['blend_skipped']})"
            if plan.get("verify_below_threshold"):
                info = plan.get("last_info") or {}
                note += (f" (verify: similarity {info.get('similarity_after')} "
                         f"below threshold "
                         f"{plan.get('verify_threshold', 0.55):.2f})")
            for i, infotext in enumerate(processed.infotexts or []):
                if "FaceConsistency:" not in infotext:
                    processed.infotexts[i] = infotext + "\n" + note


    def _on_ui_settings():
        section = (ffc_settings.SECTION_ID, ffc_settings.SECTION_LABEL)
        shared.opts.add_option(ffc_settings.OPT_ENABLED, shared.OptionInfo(
            ffc_settings.DEFAULT_ENABLED, "Face Consistency enabled by default",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_STRENGTH, shared.OptionInfo(
            ffc_settings.DEFAULT_STRENGTH, "Face Consistency default strength",
            gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_RESTORE, shared.OptionInfo(
            ffc_settings.DEFAULT_RESTORE, "Restore face detail after swap by default",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_VERIFY_THRESHOLD, shared.OptionInfo(
            ffc_settings.DEFAULT_VERIFY_THRESHOLD,
            "ArcFace similarity reported as 'same person' at or above this value",
            gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_INSWAPPER_PATH, shared.OptionInfo(
            "", "Path to inswapper_128.onnx (empty = models/insightface convention)",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_LORA_NAME, shared.OptionInfo(
            ffc_settings.DEFAULT_LORA_NAME,
            "Character LoRA name (a file in models/Lora, without extension). "
            "Appended as <lora:name:weight> when the script is enabled — "
            "this is what preserves body type as well as the face.",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_LORA_WEIGHT, shared.OptionInfo(
            ffc_settings.DEFAULT_LORA_WEIGHT,
            "Character LoRA weight (0 = off)",
            gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_OUTFIT_STRENGTH, shared.OptionInfo(
            ffc_settings.DEFAULT_OUTFIT_STRENGTH,
            "Outfit / object reference default strength (0 = off)",
            gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_TORSO_REF_DIR, shared.OptionInfo(
            ffc_settings.DEFAULT_TORSO_REF_DIR,
            "Torso reference folder: photos/crops of the person's torso "
            "(chest + waist). A square torso crop of the first readable "
            "image conditions the torso IP-Adapter unit.",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_TORSO_STRENGTH, shared.OptionInfo(
            ffc_settings.DEFAULT_TORSO_STRENGTH,
            "Torso reference default strength (0 = off). 0.45 anchors the "
            "torso without letting the adapter dominate the prompt.",
            gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_TORSO_DEPTH_WEIGHT, shared.OptionInfo(
            ffc_settings.DEFAULT_TORSO_DEPTH_WEIGHT,
            "Torso depth-lock weight (0 = off). Needs a depth ControlNet "
            "model in models/ControlNet; geometrically anchors breast "
            "volume and waist curve.",
            gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_TORSO_DEPTH_PREPROCESSOR, shared.OptionInfo(
            ffc_settings.DEFAULT_TORSO_DEPTH_PREPROCESSOR,
            "Depth preprocessor for the torso depth-lock unit",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_NAVEL_DETAILER, shared.OptionInfo(
            ffc_settings.DEFAULT_NAVEL_DETAILER,
            "Navel detailer pass: re-render the navel ROI at 3x with a "
            "low-denoise img2img pass after generation (experimental; "
            "stabilizes navel position and rendering)",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_NAVEL_GATE, shared.OptionInfo(
            ffc_settings.OPT_NAVEL_GATE,
            "Navel-position gate: template-match the navel in the waist "
            "ROI against the reference navel crop ('warn' = infotext "
            "warning; there is no reject mode by design)",
            gr.Dropdown, {"choices": list(ffc_settings.NAVEL_GATE_MODES)},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_NAVEL_TEMPLATE, shared.OptionInfo(
            ffc_settings.DEFAULT_NAVEL_TEMPLATE,
            "Path to a small reference crop of the person's navel "
            "(used by the navel gate template match)",
            section=section))
        shared.opts.add_option(ffc_settings.OPT_BODY_GATE, shared.OptionInfo(
            ffc_settings.DEFAULT_BODY_GATE,
            "Body-proportion gate: compare the generated body's proportions "
            "against the reference profile below ('warn' = infotext warning, "
            "'reject' = abort the generation before the image is presented)",
            gr.Dropdown, {"choices": list(ffc_settings.BODY_GATE_MODES)},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_BODY_TOLERANCE, shared.OptionInfo(
            ffc_settings.DEFAULT_BODY_TOLERANCE,
            "Body-proportion gate tolerance (relative deviation, 0.15 = ±15%)",
            gr.Slider, {"minimum": 0.0, "maximum": 0.5, "step": 0.01},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_REF_HEADS_TALL, shared.OptionInfo(
            ffc_settings.DEFAULT_REF_HEADS_TALL,
            "Reference body height, in head heights (e.g. 7.0)",
            gr.Slider, {"minimum": 5.0, "maximum": 9.0, "step": 0.1},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_REF_SHOULDER_HIP, shared.OptionInfo(
            ffc_settings.DEFAULT_REF_SHOULDER_HIP,
            "Reference shoulder:hip width ratio (e.g. 1.23)",
            gr.Slider, {"minimum": 0.8, "maximum": 1.6, "step": 0.01},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_REF_SHOULDER_HEADS, shared.OptionInfo(
            ffc_settings.DEFAULT_REF_SHOULDER_HEADS,
            "Reference shoulder width, in head heights (e.g. 2.2)",
            gr.Slider, {"minimum": 1.0, "maximum": 3.5, "step": 0.05},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_REF_HIP_HEADS, shared.OptionInfo(
            ffc_settings.DEFAULT_REF_HIP_HEADS,
            "Reference hip width, in head heights (e.g. 1.8)",
            gr.Slider, {"minimum": 1.0, "maximum": 3.0, "step": 0.05},
            section=section))
        shared.opts.add_option(ffc_settings.OPT_REF_LEG_FRACTION, shared.OptionInfo(
            ffc_settings.DEFAULT_REF_LEG_FRACTION,
            "Reference leg length as a fraction of body height (e.g. 0.46)",
            gr.Slider, {"minimum": 0.3, "maximum": 0.6, "step": 0.01},
            section=section))

    on_ui_settings(_on_ui_settings)
