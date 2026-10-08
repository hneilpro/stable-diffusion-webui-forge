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


def test_template_cache_key_separates_sources():
    from face_consistency.swap_engine import FaceSwapEngine

    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    assert FaceSwapEngine._template_key(a) != FaceSwapEngine._template_key(b)
    assert FaceSwapEngine._template_key("/refs") == \
        FaceSwapEngine._template_key("/refs")
