"""Offline tests for the Forge-facing script shell.

The Forge/gradio modules are stubbed before the script is imported (the
same pattern as the disabled-no-op harness), so the real
FaceConsistencyScript methods, the ControlNet slot injection, the
folder-only representative image (F1), downgrade recording, the verify
threshold warning (F4), and the blend-skip note (F8) are all exercised
without a Forge runtime, GPU, or InsightFace.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# --- stub Forge/gradio before importing the script shell ----------------
sys.modules.setdefault("gradio", MagicMock(name="gradio"))
_modules_pkg = types.ModuleType("modules")
_scripts_mod = types.ModuleType("modules.scripts")


class _FakeScript:
    pass


_scripts_mod.Script = _FakeScript
_scripts_mod.AlwaysVisible = object()
_shared_mod = types.ModuleType("modules.shared")
_shared_mod.opts = MagicMock()
_shared_mod.opts.data = {}
_cb_mod = types.ModuleType("modules.script_callbacks")
_cb_mod.on_ui_settings = lambda fn: None
_ui_mod = types.ModuleType("modules.ui_components")
_ui_mod.InputAccordion = MagicMock()
sys.modules.setdefault("modules", _modules_pkg)
sys.modules.setdefault("modules.scripts", _scripts_mod)
sys.modules.setdefault("modules.shared", _shared_mod)
sys.modules.setdefault("modules.script_callbacks", _cb_mod)
sys.modules.setdefault("modules.ui_components", _ui_mod)
_modules_pkg.scripts = _scripts_mod
_modules_pkg.shared = _shared_mod

_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_PATH = os.path.join(_EXT_DIR, "scripts", "forge_face_consistency.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("ffc_script", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()
logic = mod.logic


def _script():
    return mod.FaceConsistencyScript.__new__(mod.FaceConsistencyScript)


def _cn_runner(script_args, units_enabled):
    cn = SimpleNamespace(args_from=0, args_to=len(units_enabled))
    cn.title = lambda: "ControlNet"
    runner = SimpleNamespace(alwayson_scripts=[cn])
    return runner


def _fake_p(**kw):
    p = SimpleNamespace(extra_generation_params={}, **kw)
    return p


# --- enable OFF is a true no-op -------------------------------------------


def test_disabled_postprocess_image_leaves_image_untouched():
    inst = _script()
    sentinel = object()
    p = _fake_p(_ffc_plan={"enabled": False, "mode": "swap", "strength": 1.0})
    pp = SimpleNamespace(image=sentinel)
    assert inst.postprocess_image(p, pp, False, None, "", 1.0, True) is None
    assert pp.image is sentinel


def test_no_plan_postprocess_image_leaves_image_untouched():
    inst = _script()
    sentinel = object()
    p = _fake_p()
    pp = SimpleNamespace(image=sentinel)
    assert inst.postprocess_image(p, pp, False, None, "", 0.0, False) is None
    assert pp.image is sentinel


def test_disabled_before_process_writes_nothing():
    inst = _script()
    p = _fake_p()
    inst.before_process(p, False, None, "", 0.85, True)
    assert p._ffc_plan["enabled"] is False
    assert p._ffc_plan["mode"] == "disabled"
    assert p.extra_generation_params == {}


def test_disabled_postprocess_leaves_infotexts_unchanged():
    inst = _script()
    p = _fake_p(_ffc_plan={"enabled": False, "mode": "disabled", "strength": 0.0})
    processed = SimpleNamespace(infotexts=["hello"])
    inst.postprocess(p, processed)
    assert processed.infotexts == ["hello"]


# --- ControlNet slot injection ---------------------------------------------


def test_injection_first_disabled_slot_wins():
    unit = SimpleNamespace(enabled=True)
    args = [SimpleNamespace(enabled=True), SimpleNamespace(enabled=False),
            SimpleNamespace(enabled=False)]
    p = _fake_p(scripts=_cn_runner(args, [True, False, False]), script_args=args)
    assert mod._inject_controlnet_unit(p, unit) is True
    assert p.script_args[1] is unit
    assert p.script_args[0].enabled is True
    assert p.script_args[2].enabled is False


def test_injection_accepts_dict_units():
    unit = SimpleNamespace(enabled=True)
    args = [{"enabled": True}, {"enabled": False}]
    p = _fake_p(scripts=_cn_runner(args, [True, False]), script_args=args)
    assert mod._inject_controlnet_unit(p, unit) is True
    assert p.script_args[1] is unit


def test_injection_rewrites_tuple_script_args():
    # Live round 3 (2026-10-08): Forge hands p.script_args over as an
    # immutable tuple; assigning into it raised "'tuple' object does not
    # support item assignment" and the injection never happened.
    unit = SimpleNamespace(enabled=True)
    args = (SimpleNamespace(enabled=True), SimpleNamespace(enabled=False))
    p = _fake_p(scripts=_cn_runner(args, [True, False]), script_args=args)
    assert mod._inject_controlnet_unit(p, unit) is True
    assert isinstance(p.script_args, list)
    assert p.script_args[1] is unit
    assert p.script_args[0].enabled is True


def test_injection_all_busy_returns_false():
    unit = SimpleNamespace(enabled=True)
    args = [SimpleNamespace(enabled=True), SimpleNamespace(enabled=True)]
    p = _fake_p(scripts=_cn_runner(args, [True, True]), script_args=args)
    assert mod._inject_controlnet_unit(p, unit) is False
    assert all(a is not unit for a in p.script_args)


def test_injection_without_controlnet_script_returns_false():
    unit = SimpleNamespace(enabled=True)
    other = SimpleNamespace(args_from=0, args_to=1)
    other.title = lambda: "Something Else"
    p = _fake_p(scripts=SimpleNamespace(alwayson_scripts=[other]),
                script_args=[SimpleNamespace(enabled=False)])
    assert mod._inject_controlnet_unit(p, unit) is False


# --- F1: folder-only reference and the ControlNet unit image ---------------


def _sdxl_model(filename="model.safetensors"):
    return SimpleNamespace(
        is_sdxl=True,
        sd_checkpoint_info=SimpleNamespace(filename=filename))


def _patch_adapter(monkeypatch):
    monkeypatch.setattr(
        mod, "_available_adapters",
        lambda: (("faceid", logic.PREPROCESSOR_FACEID, "fake-faceid [abcd]"),
                 ["None", "fake-faceid [abcd]"]))


def test_folder_only_reference_builds_unit_with_real_image(monkeypatch, tmp_path):
    (tmp_path / "ref.jpg").write_bytes(b"fake")
    captured = {}

    def fake_build(pre, model_name, weight, ref_rgb):
        captured.update(image=ref_rgb, weight=weight, model=model_name)
        return SimpleNamespace(enabled=True, image=ref_rgb)

    monkeypatch.setattr(mod, "_build_controlnet_unit", fake_build)
    monkeypatch.setattr(mod, "_load_rgb_from_path",
                        lambda path: np.zeros((4, 4, 3), dtype=np.uint8))
    _patch_adapter(monkeypatch)
    args = [SimpleNamespace(enabled=False)]
    p = _fake_p(sd_model=_sdxl_model(), scripts=_cn_runner(args, [False]),
                script_args=args)
    _script().before_process(p, True, None, str(tmp_path), 0.85, True)
    assert p._ffc_plan["mode"] == "faceid"
    assert isinstance(captured["image"], np.ndarray)
    assert captured["image"].size > 0  # never np.asarray(None)
    assert captured["weight"] == pytest.approx(0.85)


def test_folder_without_images_downgrades_loudly(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(
        mod, "_build_controlnet_unit",
        lambda *a, **k: built.append(1) or SimpleNamespace(enabled=True))
    _patch_adapter(monkeypatch)
    p = _fake_p(sd_model=_sdxl_model())
    _script().before_process(p, True, None, str(tmp_path), 0.85, True)
    assert built == []  # no unit built from a missing image
    assert p._ffc_plan["mode"] == "blended-swap"
    assert p._ffc_plan["downgraded"] is True
    assert "ControlNet unit" in p._ffc_plan["reason"]
    assert "ControlNet unit" in p.extra_generation_params[
        "FaceConsistency downgrade"]


def test_representative_prefers_folder_over_single_image(monkeypatch, tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    monkeypatch.setattr(mod, "_load_rgb_from_path",
                        lambda path: np.full((2, 2, 3), 7, dtype=np.uint8))
    rgb, note = mod.representative_rgb_for_unit(
        np.zeros((2, 2, 3), dtype=np.uint8), str(tmp_path))
    assert rgb is not None and rgb[0, 0, 0] == 7
    assert "a.jpg" in note  # sorted first file wins


# --- downgrade is recorded in the generation params -------------------------


def test_flux_downgrade_recorded_in_params():
    inst = _script()
    model = SimpleNamespace(
        is_sdxl=False,
        sd_checkpoint_info=SimpleNamespace(filename="flux1-dev.safetensors"))
    p = _fake_p(sd_model=model)
    ref = np.zeros((4, 4, 3), dtype=np.uint8)
    inst.before_process(p, True, ref, "", 0.85, True)
    assert p._ffc_plan["mode"] == "blended-swap"
    assert p._ffc_plan["downgraded"] is True
    assert "Flux" in p.extra_generation_params["FaceConsistency downgrade"]


# --- F4/F8: verify warning and blend-skip are never silent -------------------


def _fake_cv2():
    fake = types.ModuleType("cv2")
    fake.COLOR_RGB2BGR = 0
    fake.COLOR_BGR2RGB = 1
    fake.cvtColor = lambda arr, code: np.ascontiguousarray(arr)
    return fake


def _fake_pil():
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.fromarray = lambda arr: ("pil-image", arr.shape)
    pil.Image = image_mod
    return pil, image_mod


def _run_blended(monkeypatch, faces, similarity_after):
    pil, image_mod = _fake_pil()
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)
    info = {"similarity_before": 0.10, "similarity_after": similarity_after,
            "restored_by": "detail-graft"}

    class _App:
        def get(self, img):
            return faces

    engine = SimpleNamespace(
        app=_App(),
        swap=lambda target, ref, restore=True: (np.array(target, copy=True), info))
    monkeypatch.setattr(mod, "_get_engine", lambda p: engine)
    plan = {"enabled": True, "mode": "blended-swap", "strength": 0.5,
            "restore": True, "verify_threshold": 0.55, "refs_dir": "",
            "ref_bgr": np.zeros((8, 8, 3), dtype=np.uint8),
            "downgraded": False, "reason": "test"}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((8, 8, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.5, True)
    return p, plan


def test_blend_skipped_is_logged_and_recorded(monkeypatch):
    p, plan = _run_blended(monkeypatch, faces=[], similarity_after=0.93)
    assert "blend_skipped" in plan
    assert "FaceConsistency blend skipped" in p.extra_generation_params


def test_below_threshold_similarity_warns(monkeypatch):
    face = SimpleNamespace(bbox=(1, 1, 7, 7))
    p, plan = _run_blended(monkeypatch, faces=[face], similarity_after=0.20)
    assert plan.get("verify_below_threshold") is True
    assert p.extra_generation_params["FaceConsistency verify"].startswith(
        "below threshold")


def test_good_similarity_does_not_warn(monkeypatch):
    face = SimpleNamespace(bbox=(1, 1, 7, 7))
    p, plan = _run_blended(monkeypatch, faces=[face], similarity_after=0.93)
    assert not plan.get("verify_below_threshold")
    assert "FaceConsistency verify" not in p.extra_generation_params


# --- pure helpers behind F1/F4/F5 --------------------------------------------


def test_pick_representative_ref(tmp_path):
    assert logic.pick_representative_ref("") is None
    assert logic.pick_representative_ref(str(tmp_path)) is None
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    picked = logic.pick_representative_ref(str(tmp_path))
    assert picked.endswith("a.jpg")


def test_is_below_threshold():
    assert logic.is_below_threshold(0.20, 0.55) is True
    assert logic.is_below_threshold(0.93, 0.55) is False
    assert logic.is_below_threshold(None, 0.55) is False


def test_list_ref_candidates_sorted_and_filtered(tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    candidates = logic.list_ref_candidates(str(tmp_path))
    assert [os.path.basename(c) for c in candidates] == ["a.jpg", "b.png"]
    assert logic.list_ref_candidates("") == []
    assert logic.list_ref_candidates(str(tmp_path / "missing")) == []


def test_representative_skips_corrupt_first_file(monkeypatch, tmp_path):
    # N2: the first sorted folder file is unreadable; the next one must
    # be tried instead of giving up on the folder.
    (tmp_path / "a.jpg").write_bytes(b"corrupt")
    (tmp_path / "b.png").write_bytes(b"ok")

    def fake_load(path):
        if path.endswith("a.jpg"):
            return None
        return np.full((2, 2, 3), 9, dtype=np.uint8)

    monkeypatch.setattr(mod, "_load_rgb_from_path", fake_load)
    rgb, note = mod.representative_rgb_for_unit(None, str(tmp_path))
    assert rgb is not None and rgb[0, 0, 0] == 9
    assert "b.png" in note
    assert "skipped unreadable" in note and "a.jpg" in note


def test_representative_all_corrupt_folder_returns_none(monkeypatch, tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"corrupt")
    monkeypatch.setattr(mod, "_load_rgb_from_path", lambda path: None)
    rgb, note = mod.representative_rgb_for_unit(None, str(tmp_path))
    assert rgb is None
    assert note == "no readable image in reference folder"


# --- API string reference (live failure 2026-10-07) -------------------------
#
# The WebUI hands the script a numpy array, but the API delivers image
# script args as base64 strings. The first live run died on both paths
# with "tuple index out of range" (np.asarray(str) is a 0-d array) and
# silently generated with no swap and no injection.


def _b64_png(rgb=(10, 20, 30), size=(6, 5)):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_ref_to_rgb_decodes_base64_string():
    rgb = mod._ref_to_rgb(_b64_png())
    assert rgb is not None
    assert rgb.shape == (5, 6, 3)
    assert tuple(rgb[0, 0]) == (10, 20, 30)


def test_ref_to_rgb_decodes_data_uri_and_path(tmp_path):
    b64 = _b64_png(rgb=(1, 2, 3))
    rgb = mod._ref_to_rgb("data:image/png;base64," + b64)
    assert rgb is not None and tuple(rgb[0, 0]) == (1, 2, 3)
    path = tmp_path / "ref.png"
    import base64

    path.write_bytes(base64.b64decode(b64))
    rgb = mod._ref_to_rgb(str(path))
    assert rgb is not None and tuple(rgb[0, 0]) == (1, 2, 3)


def test_ref_to_rgb_undecodable_string_returns_none_not_crash():
    assert mod._ref_to_rgb("not an image at all") is None
    assert mod._ref_to_rgb("") is None
    assert mod._ref_to_bgr("not an image at all") is None


def test_ref_to_bgr_flips_channels_from_string():
    bgr = mod._ref_to_bgr(_b64_png(rgb=(10, 20, 30)))
    assert bgr is not None and tuple(bgr[0, 0]) == (30, 20, 10)


def test_representative_from_base64_single_image():
    rgb, note = mod.representative_rgb_for_unit(_b64_png(), "")
    assert rgb is not None and rgb.shape == (5, 6, 3)
    assert note == "single image"


def test_before_process_string_ref_injects_controlnet_unit(monkeypatch):
    captured = {}

    def fake_build(pre, model_name, weight, ref_rgb):
        captured.update(image=ref_rgb, weight=weight)
        return SimpleNamespace(enabled=True, image=ref_rgb)

    monkeypatch.setattr(mod, "_build_controlnet_unit", fake_build)
    _patch_adapter(monkeypatch)
    args = [SimpleNamespace(enabled=False)]
    p = _fake_p(sd_model=_sdxl_model(), scripts=_cn_runner(args, [False]),
                script_args=args)
    _script().before_process(p, True, _b64_png(), "", 0.85, True)
    assert p._ffc_plan["mode"] == "faceid"
    assert isinstance(captured["image"], np.ndarray)
    assert captured["image"].shape == (5, 6, 3)


def test_before_process_string_ref_at_max_sets_swap_ref():
    p = _fake_p(sd_model=_sdxl_model())
    _script().before_process(p, True, _b64_png(), "", 1.0, True)
    assert p._ffc_plan["mode"] == "swap"
    ref_bgr = p._ffc_plan["ref_bgr"]
    assert isinstance(ref_bgr, np.ndarray) and ref_bgr.shape == (5, 6, 3)


# --- setup failure falls back loudly, never raises (live round 2) -----------
#
# Live round 2 (2026-10-08): SDXL swap worked, but the below-max
# ControlNet path still produced mode=disabled with no explanation —
# something in the setup block raised and the host generation continued
# with the plan stuck at its initial value. Setup failures must land in
# the infotext and downgrade to blended swap instead of vanishing.


def test_controlnet_build_failure_downgrades_not_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unit build exploded")

    monkeypatch.setattr(mod, "_build_controlnet_unit", boom)
    _patch_adapter(monkeypatch)
    args = [SimpleNamespace(enabled=False)]
    p = _fake_p(sd_model=_sdxl_model(), scripts=_cn_runner(args, [False]),
                script_args=args)
    _script().before_process(p, True, _b64_png(), "", 0.85, True)
    assert p._ffc_plan["mode"] == "blended-swap"
    assert p._ffc_plan["downgraded"] is True
    assert "unit build exploded" in p.extra_generation_params[
        "FaceConsistency downgrade"]
    assert "unit build exploded" in p.extra_generation_params[
        "FaceConsistency error"]
    assert p.extra_generation_params["FaceConsistency mode"] == "blended-swap"


def test_family_detection_failure_downgrades_not_raises(monkeypatch):
    def boom(p):
        raise RuntimeError("family probe exploded")

    monkeypatch.setattr(mod, "_detect_family_forge", boom)
    p = _fake_p(sd_model=_sdxl_model())
    _script().before_process(p, True, _b64_png(), "", 0.85, True)
    assert p._ffc_plan["mode"] == "blended-swap"
    assert "family probe exploded" in p.extra_generation_params[
        "FaceConsistency error"]


def test_template_cache_key_separates_sources():
    from face_consistency.swap_engine import FaceSwapEngine

    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    assert FaceSwapEngine._template_key(a) != FaceSwapEngine._template_key(b)
    assert FaceSwapEngine._template_key("/refs") == \
        FaceSwapEngine._template_key("/refs")


# --- character LoRA prompt injection (settings-only, body-type lever) -------


def _patch_lora(monkeypatch, name="tori_xl", weight=0.7):
    monkeypatch.setattr(
        mod, "_opt",
        lambda n, d: {"ffc_lora_name": name,
                      "ffc_lora_weight": weight}.get(n, d))


def test_character_lora_injected_into_prompt(monkeypatch):
    _patch_lora(monkeypatch)
    p = _fake_p(sd_model=_sdxl_model(), prompt="a portrait",
                all_prompts=["a portrait"])
    _script().before_process(p, True, _b64_png(), "", 1.0, True)
    assert "<lora:tori_xl:0.7>" in p.prompt
    assert p.all_prompts == ["a portrait <lora:tori_xl:0.7>"]
    assert p.extra_generation_params["FaceConsistency lora"] == \
        "<lora:tori_xl:0.7>"


def test_character_lora_dedup_when_already_present(monkeypatch):
    _patch_lora(monkeypatch)
    p = _fake_p(sd_model=_sdxl_model(),
                prompt="a portrait <lora:tori_xl:0.5>",
                all_prompts=["a portrait <lora:tori_xl:0.5>"])
    _script().before_process(p, True, _b64_png(), "", 1.0, True)
    assert p.prompt == "a portrait <lora:tori_xl:0.5>"
    assert p.prompt.count("<lora:tori_xl:") == 1


def test_character_lora_off_by_default():
    p = _fake_p(sd_model=_sdxl_model(), prompt="a portrait",
                all_prompts=["a portrait"])
    _script().before_process(p, True, _b64_png(), "", 1.0, True)
    assert p.prompt == "a portrait"
    assert "FaceConsistency lora" not in p.extra_generation_params


def test_character_lora_zero_weight_is_noop(monkeypatch):
    _patch_lora(monkeypatch, weight=0.0)
    p = _fake_p(sd_model=_sdxl_model(), prompt="a portrait",
                all_prompts=["a portrait"])
    _script().before_process(p, True, _b64_png(), "", 1.0, True)
    assert p.prompt == "a portrait"


# --- engine-aware representative selection ----------------------------------


def _two_file_folder(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"blurry")
    (tmp_path / "b.png").write_bytes(b"sharp")
    return {
        str(tmp_path / "a.jpg"): np.full((4, 4, 3), 10, dtype=np.uint8),
        str(tmp_path / "b.png"): np.full((4, 4, 3), 200, dtype=np.uint8),
    }


def test_representative_with_engine_picks_sharpest(monkeypatch, tmp_path):
    rgbs = _two_file_folder(tmp_path)
    monkeypatch.setattr(mod, "_load_rgb_from_path", lambda path: rgbs[path])
    # sharpness keyed on pixel value: 200 (b.png) beats 10 (a.jpg)
    monkeypatch.setattr("face_consistency.swap_engine.face_sharpness",
                        lambda img, face: float(img[0, 0, 0]))
    engine = SimpleNamespace(
        detect=lambda bgr: [SimpleNamespace(bbox=(0, 0, 4, 4))])
    rgb, note = mod.representative_rgb_for_unit(
        None, str(tmp_path), engine=engine)
    assert rgb[0, 0, 0] == 200
    assert "b.png" in note and "sharpest face" in note


def test_representative_with_engine_skips_faceless(monkeypatch, tmp_path):
    rgbs = _two_file_folder(tmp_path)
    monkeypatch.setattr(mod, "_load_rgb_from_path", lambda path: rgbs[path])
    monkeypatch.setattr("face_consistency.swap_engine.face_sharpness",
                        lambda img, face: float(img[0, 0, 0]))

    def detect(bgr):
        # a.jpg (value 10) has no detectable face
        if bgr[0, 0, 0] == 10:
            return []
        return [SimpleNamespace(bbox=(0, 0, 4, 4))]

    engine = SimpleNamespace(detect=detect)
    rgb, note = mod.representative_rgb_for_unit(
        None, str(tmp_path), engine=engine)
    assert rgb[0, 0, 0] == 200
    assert "no face" in note and "a.jpg" in note


def test_representative_engine_failure_falls_back_to_first(monkeypatch,
                                                          tmp_path):
    rgbs = _two_file_folder(tmp_path)
    monkeypatch.setattr(mod, "_load_rgb_from_path", lambda path: rgbs[path])

    def boom(bgr):
        raise RuntimeError("no insightface here")

    engine = SimpleNamespace(detect=boom)
    rgb, note = mod.representative_rgb_for_unit(
        None, str(tmp_path), engine=engine)
    # sorted-first decodable file wins, exactly like engine=None
    assert rgb[0, 0, 0] == 10
    assert "a.jpg" in note


# --- outfit / object reference via built-in IP-Adapter ----------------------

def _patch_ipadapter_names(monkeypatch, names):
    monkeypatch.setattr(mod, "_available_adapters", lambda: (None, names))


def _outfit_p(sd_model, n_slots=2):
    args = [SimpleNamespace(enabled=False) for _ in range(n_slots)]
    return _fake_p(sd_model=sd_model,
                   scripts=_cn_runner(args, [False] * n_slots),
                   script_args=args)


def test_outfit_reference_injects_ipadapter_unit(monkeypatch):
    built = []

    def fake_build(pre, model_name, weight, ref_rgb):
        built.append((pre, model_name, weight))
        return SimpleNamespace(enabled=True)

    monkeypatch.setattr(mod, "_build_controlnet_unit", fake_build)
    _patch_ipadapter_names(monkeypatch, ["None", "ip-adapter_sdxl [abcd]"])
    p = _outfit_p(_sdxl_model())
    outfit = np.zeros((8, 8, 3), dtype=np.uint8)
    _script().before_process(p, True, None, "", 1.0, True, outfit, 0.6)
    assert p._ffc_plan["mode"] == "outfit-only"  # no face ref given
    assert len(built) == 1
    pre, model_name, weight = built[0]
    assert pre == logic.PREPROCESSOR_IPADAPTER_BIGG
    assert model_name == "ip-adapter_sdxl [abcd]"
    assert weight == pytest.approx(0.6)
    assert "ip-adapter_sdxl [abcd]" in p.extra_generation_params[
        "FaceConsistency outfit ref"]


def test_outfit_uses_vith_preprocessor_for_vith_model(monkeypatch):
    built = []

    def fake_build(pre, model_name, weight, ref_rgb):
        built.append(pre)
        return SimpleNamespace(enabled=True)

    monkeypatch.setattr(mod, "_build_controlnet_unit", fake_build)
    _patch_ipadapter_names(
        monkeypatch, ["None", "ip-adapter-plus_sdxl_vit-h [efgh]"])
    p = _outfit_p(_sdxl_model())
    _script().before_process(p, True, None, "", 1.0, True,
                             np.zeros((8, 8, 3), dtype=np.uint8), 0.5)
    assert built == [logic.PREPROCESSOR_IPADAPTER_H]


def test_outfit_skipped_loudly_without_model(monkeypatch):
    _patch_ipadapter_names(monkeypatch, ["None", "control_v11p_sd15_canny"])
    p = _outfit_p(_sdxl_model())
    _script().before_process(p, True, None, "", 1.0, True,
                             np.zeros((8, 8, 3), dtype=np.uint8), 0.6)
    assert "no general IP-Adapter model" in p.extra_generation_params[
        "FaceConsistency outfit ref"]


def test_outfit_skip_names_face_only_variants(monkeypatch):
    _patch_ipadapter_names(
        monkeypatch, ["None", "ip-adapter-plus-face_sdxl_vit-h [368cf551]"])
    p = _outfit_p(_sdxl_model())
    _script().before_process(p, True, None, "", 1.0, True,
                             np.zeros((8, 8, 3), dtype=np.uint8), 0.6)
    note = p.extra_generation_params["FaceConsistency outfit ref"]
    assert "no general IP-Adapter model" in note
    assert "ip-adapter-plus-face_sdxl_vit-h [368cf551]" in note


def test_outfit_skipped_on_flux():
    model = SimpleNamespace(
        is_sdxl=False,
        sd_checkpoint_info=SimpleNamespace(filename="flux1-dev.safetensors"))
    p = _fake_p(sd_model=model)
    _script().before_process(p, True, None, "", 1.0, True,
                             np.zeros((8, 8, 3), dtype=np.uint8), 0.6)
    assert "Flux" in p.extra_generation_params["FaceConsistency outfit ref"]


def test_face_and_outfit_units_share_slots(monkeypatch):
    built = []

    def fake_build(pre, model_name, weight, ref_rgb):
        built.append((pre, model_name, weight))
        return SimpleNamespace(enabled=True)

    monkeypatch.setattr(mod, "_build_controlnet_unit", fake_build)
    monkeypatch.setattr(
        mod, "_available_adapters",
        lambda: (("faceid", logic.PREPROCESSOR_FACEID, "fake-faceid [a]"),
                 ["None", "fake-faceid [a]", "ip-adapter_sdxl [b]"]))
    args = [SimpleNamespace(enabled=False), SimpleNamespace(enabled=False)]
    p = _fake_p(sd_model=_sdxl_model(),
                scripts=_cn_runner(args, [False, False]),
                script_args=args)
    face = np.zeros((8, 8, 3), dtype=np.uint8)
    outfit = np.ones((8, 8, 3), dtype=np.uint8)
    _script().before_process(p, True, face, "", 0.85, True, outfit, 0.6)
    assert p._ffc_plan["mode"] == "faceid"
    assert len(built) == 2
    assert built[0][0] == logic.PREPROCESSOR_FACEID
    assert built[0][2] == pytest.approx(0.85)
    assert built[1][0] == logic.PREPROCESSOR_IPADAPTER_BIGG
    assert built[1][2] == pytest.approx(0.6)
    assert p.script_args[0].enabled is True
    assert p.script_args[1].enabled is True


def test_outfit_off_by_default_writes_nothing(monkeypatch):
    built = []
    monkeypatch.setattr(
        mod, "_build_controlnet_unit",
        lambda *a, **k: built.append(1) or SimpleNamespace(enabled=True))
    _patch_ipadapter_names(monkeypatch, ["None", "ip-adapter_sdxl [abcd]"])
    p = _outfit_p(_sdxl_model())
    # strength 0 -> outfit path idle, no infotext key
    _script().before_process(p, True, None, "", 1.0, True,
                             np.zeros((8, 8, 3), dtype=np.uint8), 0.0)
    assert built == []
    assert "FaceConsistency outfit ref" not in p.extra_generation_params


# --- head-height proportion gate in postprocess_image -------------------------

def _run_blended_with_bbox(monkeypatch, img_hw, bbox, similarity_after):
    pil, image_mod = _fake_pil()
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)
    info = {"similarity_before": 0.10, "similarity_after": similarity_after,
            "restored_by": "detail-graft",
            "face_bbox": [float(v) for v in bbox],
            "face_size_ratio": 0.05}

    class _App:
        def get(self, img):
            return [SimpleNamespace(bbox=bbox)]

    engine = SimpleNamespace(
        app=_App(),
        swap=lambda target, ref, restore=True: (np.array(target, copy=True),
                                                info))
    monkeypatch.setattr(mod, "_get_engine", lambda p: engine)
    plan = {"enabled": True, "mode": "blended-swap", "strength": 0.5,
            "restore": True, "verify_threshold": 0.55, "refs_dir": "",
            "ref_bgr": np.zeros((8, 8, 3), dtype=np.uint8),
            "downgraded": False, "reason": "test"}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((img_hw, img_hw, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.5, True)
    return p


def test_head_height_recorded_in_params(monkeypatch):
    p = _run_blended_with_bbox(monkeypatch, 64, (10, 10, 30, 30), 0.93)
    assert p.extra_generation_params[
        "FaceConsistency head height"] == "31.2% of frame"
    assert "FaceConsistency proportion" not in p.extra_generation_params


def test_head_height_warns_when_tiny(monkeypatch):
    p = _run_blended_with_bbox(monkeypatch, 64, (1, 1, 5, 5), 0.93)
    assert "closer crop" in p.extra_generation_params[
        "FaceConsistency proportion"]


# --- body-proportion gate (shell integration) ------------------------------

_PASS_KP = {
    "nose": (400.0, 150.0), "neck": (400.0, 200.0),
    "shoulder_r": (510.0, 210.0), "shoulder_l": (290.0, 210.0),
    "hip_r": (490.0, 480.0), "hip_l": (310.0, 480.0),
    "ankle_r": (485.0, 802.0), "ankle_l": (315.0, 802.0),
}
_FAIL_KP = {
    "nose": (400.0, 150.0), "neck": (400.0, 200.0),
    "shoulder_r": (520.0, 210.0), "shoulder_l": (280.0, 210.0),
    "hip_r": (530.0, 480.0), "hip_l": (270.0, 480.0),  # 2.6 heads: +44%
    "ankle_r": (525.0, 850.0), "ankle_l": (275.0, 850.0),
}
_GATE_OPTS = {
    "ffc_body_gate": "warn",
    "ffc_body_tolerance": 0.15,
    "ffc_ref_heads_tall": 7.0,
    "ffc_ref_shoulder_hip": 1.23,
    "ffc_ref_shoulder_heads": 2.2,
    "ffc_ref_hip_heads": 1.8,
    "ffc_ref_leg_fraction": 0.46,
}


def _run_faceid_with_gate(monkeypatch, gate_overrides, keypoints,
                          head_bbox=(0.0, 0.0, 80.0, 100.0)):
    """postprocess_image in a non-swap mode so only the gate runs."""
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    opts = dict(_GATE_OPTS)
    opts.update(gate_overrides)
    monkeypatch.setattr(mod, "_opt", lambda name, default: opts.get(name, default))
    monkeypatch.setattr(mod, "_detect_body_keypoints",
                        lambda p, bgr: keypoints)
    monkeypatch.setattr(mod, "_head_bbox_for_gate",
                        lambda p, bgr: head_bbox)
    plan = {"enabled": True, "mode": "faceid", "strength": 0.85,
            "restore": True, "verify_threshold": 0.55,
            "downgraded": False, "reason": "test"}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.85, True)
    return p


def test_body_gate_off_is_noop(monkeypatch):
    def _boom(p, bgr):
        raise AssertionError("provider must not run when gate is off")

    monkeypatch.setattr(mod, "_opt",
                        lambda name, default: {"ffc_body_gate": "off"}.get(name, default))
    monkeypatch.setattr(mod, "_detect_body_keypoints", _boom)
    plan = {"enabled": True, "mode": "faceid", "strength": 0.85}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.85, True)
    assert not any(k.startswith("FaceConsistency body")
                   for k in p.extra_generation_params)


def test_body_gate_warn_records_infotext(monkeypatch):
    p = _run_faceid_with_gate(monkeypatch, {"ffc_body_gate": "warn"}, _FAIL_KP)
    assert "FaceConsistency body" in p.extra_generation_params
    warning = p.extra_generation_params["FaceConsistency body gate"]
    assert warning.startswith("WARNING")
    assert "hip width" in warning


def test_body_gate_pass_records_measurements(monkeypatch):
    p = _run_faceid_with_gate(monkeypatch, {"ffc_body_gate": "warn"}, _PASS_KP)
    assert "7.02 heads tall" in p.extra_generation_params["FaceConsistency body"]
    assert "FaceConsistency body gate" not in p.extra_generation_params


def test_body_gate_reject_raises(monkeypatch):
    import sys

    from face_consistency import body_gate as body_gate_mod

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setattr(mod, "_opt", lambda name, default: {
        **_GATE_OPTS, "ffc_body_gate": "reject"}.get(name, default))
    monkeypatch.setattr(mod, "_detect_body_keypoints",
                        lambda p, bgr: _FAIL_KP)
    monkeypatch.setattr(mod, "_head_bbox_for_gate",
                        lambda p, bgr: (0.0, 0.0, 80.0, 100.0))
    plan = {"enabled": True, "mode": "faceid", "strength": 0.85}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(body_gate_mod.BodyProportionError):
        _script().postprocess_image(p, pp, True, None, "", 0.85, True)
    assert "REJECTED" in p.extra_generation_params["FaceConsistency body gate"]


def test_body_gate_no_pose_degrades_loudly(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setattr(mod, "_opt",
                        lambda name, default: _GATE_OPTS.get(name, default))
    # Detector present but finds no pose.
    monkeypatch.setattr(mod, "_get_pose_detector", lambda p: (lambda bgr: []))
    plan = {"enabled": True, "mode": "faceid", "strength": 0.85}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.85, True)
    assert "no pose detected" in p.extra_generation_params[
        "FaceConsistency body gate"]


def test_body_gate_detector_unavailable_degrades_loudly(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setattr(mod, "_opt",
                        lambda name, default: _GATE_OPTS.get(name, default))
    monkeypatch.setattr(mod, "_get_pose_detector", lambda p: None)
    plan = {"enabled": True, "mode": "faceid", "strength": 0.85}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.85, True)
    assert "pose detector unavailable" in p.extra_generation_params[
        "FaceConsistency body gate"]


def test_body_gate_skipped_when_mode_disabled(monkeypatch):
    def _boom(p, bgr):
        raise AssertionError("gate must not run in disabled mode")

    monkeypatch.setattr(mod, "_opt",
                        lambda name, default: {"ffc_body_gate": "reject"}.get(name, default))
    monkeypatch.setattr(mod, "_detect_body_keypoints", _boom)
    plan = {"enabled": True, "mode": "disabled", "strength": 0.0}
    p = _fake_p(_ffc_plan=plan)
    pp = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
    _script().postprocess_image(p, pp, True, None, "", 0.0, True)
    assert not any(k.startswith("FaceConsistency body")
                   for k in p.extra_generation_params)


def test_body_gate_settings_registered(monkeypatch):
    added = {}

    class _Opts:
        def add_option(self, name, info):
            added[name] = info

    monkeypatch.setattr(mod.shared, "opts",
                        SimpleNamespace(add_option=_Opts().add_option, data={}))
    monkeypatch.setattr(mod.shared, "OptionInfo",
                        lambda *a, **k: ("OptionInfo", a, k),
                        raising=False)
    mod._on_ui_settings()
    for name in ("ffc_body_gate", "ffc_body_tolerance", "ffc_ref_heads_tall",
                 "ffc_ref_shoulder_hip", "ffc_ref_shoulder_heads",
                 "ffc_ref_hip_heads", "ffc_ref_leg_fraction"):
        assert name in added, name


# --- navel detailer UI toggle (9th arg) ----------------------------------


def test_ui_returns_nine_args_with_navel_toggle():
    inst = _script()
    assert len(inst.ui(False)) == 9


def test_before_process_navel_toggle_explicit_true():
    inst = _script()
    p = _fake_p()
    inst.before_process(p, False, None, "", 0.85, True, None, 0.0, 0.0, True)
    assert p._ffc_navel_detailer is True


def test_before_process_navel_toggle_defaults_to_settings_off():
    inst = _script()
    p = _fake_p()
    inst.before_process(p, False, None, "", 0.85, True)
    assert p._ffc_navel_detailer is False


def test_postprocess_image_accepts_nine_args():
    inst = _script()
    sentinel = object()
    p = _fake_p(_ffc_plan={"enabled": False, "mode": "swap", "strength": 1.0})
    pp = SimpleNamespace(image=sentinel)
    assert inst.postprocess_image(p, pp, False, None, "", 1.0, True,
                                  None, 0.0, 0.0, True) is None
    assert pp.image is sentinel
