"""Forge Face Consistency — reference strength with max = face swap.

One control (Enable + reference image + Strength) keeps a chosen face
consistent across SDXL and Flux checkpoints:

- Strength = max (1.0): full InsightFace swap + restore in
  postprocess_image. Model-agnostic; works for SDXL and Flux.
- Strength < max on SDXL with a FaceID/InstantID ControlNet model present:
  inject a ControlNet unit at weight=strength (in-diffusion reference).
- Otherwise (Flux, or SDXL without an adapter): blended swap at
  blend=strength inside a feathered face mask.

Every run records the mode actually used, the strength, and the ArcFace
similarity before/after in the infotext. A downgrade is always logged,
never silent. Flux never receives SDXL adapters.

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


def _detect_family_forge(p) -> str:
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
    return logic.detect_family(is_sdxl=is_sdxl, checkpoint_hint=hint, class_name=cls)


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


def representative_rgb_for_unit(ref_image, refs_dir):
    """Concrete RGB image for a ControlNet unit.

    The unit conditions on one image, so a folder-only reference needs
    a representative file decoded here (the swap template path still
    uses the full folder). A non-empty folder overrides the single
    image, matching the UI wording. Returns (rgb_or_None, source_note).
    """
    folder = str(refs_dir).strip() if refs_dir else ""
    if folder:
        # N2: try every folder file in sorted order until one decodes;
        # a corrupt first file must not discard a usable folder.
        skipped = []
        for rep_path in logic.list_ref_candidates(folder):
            rgb = _load_rgb_from_path(rep_path)
            if rgb is not None and rgb.size:
                note = f"folder representative {rep_path}"
                if skipped:
                    note += f" (skipped unreadable: {', '.join(skipped)})"
                return rgb, note
            skipped.append(rep_path)
        if ref_image is not None:
            rgb = _ref_to_rgb(ref_image)
            if rgb is not None and rgb.size:
                return rgb, "single image (folder representative unreadable)"
        return None, "no readable image in reference folder"
    rgb = _ref_to_rgb(ref_image)
    if rgb is not None and rgb.size:
        return rgb, "single image"
    return None, "no reference image"


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
    script_args = getattr(p, "script_args", None)
    if not script_args:
        return False
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
            self.infotext_fields = [
                (enabled, "FaceConsistency enabled"),
                (strength, "FaceConsistency strength"),
            ]
            return enabled, ref_image, refs_dir, strength, restore

        # -- decision + ControlNet injection (runs before sampling) -------
        def before_process(self, p, enabled, ref_image, refs_dir, strength, restore):
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
            if ref_image is None and not (refs_dir and str(refs_dir).strip()):
                plan["reason"] = "enabled but no reference image or folder given"
                _log("enabled with no reference; doing nothing")
                return
            try:
                family = _detect_family_forge(p)
                adapter, _names = _available_adapters()
                mode, downgraded, reason = logic.decide_mode(
                    plan["strength"], family, adapter is not None)
                plan.update(family=family, downgraded=downgraded, reason=reason)
                if mode == "controlnet" and adapter is not None:
                    kind, preprocessor_name, model_name = adapter
                    # F1: the unit needs a concrete image. A folder-only (or
                    # folder-overriding) reference must hand ControlNet a
                    # decoded representative, never np.asarray(None).
                    unit_rgb, unit_src = representative_rgb_for_unit(
                        ref_image, refs_dir)
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
                if plan["mode"] in ("swap", "blended-swap"):
                    plan["ref_bgr"] = _ref_to_bgr(ref_image)
                    plan["refs_dir"] = str(refs_dir).strip() if refs_dir else ""
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
                plan["ref_bgr"] = _ref_to_bgr(ref_image)
                plan["refs_dir"] = str(refs_dir).strip() if refs_dir else ""
                plan.update(mode="blended-swap", downgraded=True,
                            reason=f"setup failed ({exc}); using blended swap")
                _log(f"setup FAILED ({exc}); falling back to blended swap")
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
        def postprocess_image(self, p, pp, enabled, ref_image, refs_dir, strength, restore):
            plan = getattr(p, "_ffc_plan", None)
            if not plan or not plan.get("enabled"):
                return
            if plan["mode"] not in ("swap", "blended-swap"):
                # ControlNet reference path: identity was formed in
                # diffusion, so there is no swap similarity to report.
                # Record n/a explicitly instead of staying silent (F7).
                p.extra_generation_params.setdefault(
                    "FaceConsistency similarity before", "n/a (controlnet)")
                p.extra_generation_params.setdefault(
                    "FaceConsistency similarity after", "n/a (controlnet)")
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

    on_ui_settings(_on_ui_settings)
