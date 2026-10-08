"""Offline tests for the body-proportion gate (pure logic, no Forge)."""

from types import SimpleNamespace

import pytest

from face_consistency import body_gate

REF = {
    "heads_tall": 7.0,
    "shoulder_hip": 1.23,
    "shoulder_heads": 2.2,
    "hip_heads": 1.8,
    "leg_fraction": 0.46,
}
TOL = 0.15
HEAD_H = 100.0


def _vj_coco18():
    """COCO-18 keypoints matching the reference profile.

    Head height 100px; shoulders 220px wide (2.2 heads); hips 180px
    (1.8 heads); body 702px tall (7.02 heads); legs ~322px (45.9%).
    """
    kp = [(0.0, 0.0, 0.0)] * 18
    kp = [list(p) for p in kp]
    kp[0] = [400.0, 150.0, 0.9]   # nose
    kp[1] = [400.0, 200.0, 0.9]   # neck
    kp[2] = [510.0, 210.0, 0.9]   # shoulder_r
    kp[5] = [290.0, 210.0, 0.9]   # shoulder_l
    kp[8] = [490.0, 480.0, 0.9]   # hip_r
    kp[11] = [310.0, 480.0, 0.9]  # hip_l
    kp[9] = [485.0, 640.0, 0.9]   # knee_r
    kp[12] = [315.0, 640.0, 0.9]  # knee_l
    kp[10] = [485.0, 802.0, 0.9]  # ankle_r
    kp[13] = [315.0, 802.0, 0.9]  # ankle_l
    return kp


def _exaggerated_coco18():
    """Same skeleton, but a much curvier/wider body than the reference."""
    kp = _vj_coco18()
    kp[2] = [520.0, 210.0, 0.9]
    kp[5] = [280.0, 210.0, 0.9]   # shoulders 240px
    kp[8] = [530.0, 480.0, 0.9]
    kp[11] = [270.0, 480.0, 0.9]  # hips 260px (hip_heads 2.6, +44%)
    kp[10] = [525.0, 850.0, 0.9]
    kp[13] = [275.0, 850.0, 0.9]
    return kp


# --- normalize_keypoints ------------------------------------------------


def test_normalize_coco18_picks_right_indices():
    kp = body_gate.normalize_keypoints(_vj_coco18())
    assert kp["nose"] == (400.0, 150.0)
    assert kp["shoulder_r"] == (510.0, 210.0)
    assert kp["hip_l"] == (310.0, 480.0)
    assert kp["ankle_r"] == (485.0, 802.0)


def test_normalize_body25_uses_its_own_hip_indices():
    raw = [(0.0, 0.0, 0.0)] * 25
    raw = [list(p) for p in raw]
    raw[9] = [111.0, 222.0, 0.9]    # BODY_25 hip_r
    raw[12] = [333.0, 444.0, 0.9]   # BODY_25 hip_l
    raw[8] = [999.0, 999.0, 0.9]    # COCO hip_r -- must be ignored
    kp = body_gate.normalize_keypoints(raw)
    assert kp["hip_r"] == (111.0, 222.0)
    assert kp["hip_l"] == (333.0, 444.0)


def test_normalize_drops_low_score_keypoints():
    raw = _vj_coco18()
    raw[2] = [510.0, 210.0, 0.05]  # shoulder_r below SCORE_MIN
    kp = body_gate.normalize_keypoints(raw)
    assert "shoulder_r" not in kp
    assert "shoulder_l" in kp


def test_normalize_accepts_object_keypoints():
    raw = [SimpleNamespace(x=1.0, y=2.0, score=0.9) if i == 0
           else (0.0, 0.0, 0.0) for i in range(18)]
    kp = body_gate.normalize_keypoints(raw)
    assert kp["nose"] == (1.0, 2.0)


def test_normalize_empty_returns_empty():
    assert body_gate.normalize_keypoints([]) == {}
    assert body_gate.normalize_keypoints(None) == {}


# --- measure_ratios / assess --------------------------------------------


def test_measure_vj_profile_passes():
    kp = body_gate.normalize_keypoints(_vj_coco18())
    measured = body_gate.measure_ratios(kp, HEAD_H)
    assert set(measured) == {"heads_tall", "shoulder_hip", "shoulder_heads",
                             "hip_heads", "leg_fraction"}
    assert measured["heads_tall"] == pytest.approx(7.02, abs=0.01)
    assert measured["shoulder_hip"] == pytest.approx(1.2222, abs=0.001)
    assert measured["shoulder_heads"] == pytest.approx(2.2, abs=0.01)
    assert measured["hip_heads"] == pytest.approx(1.8, abs=0.01)
    assert measured["leg_fraction"] == pytest.approx(0.4588, abs=0.002)
    verdict, details = body_gate.assess(measured, REF, TOL)
    assert verdict == "pass"
    assert all(d["ok"] for d in details.values())


def test_measure_exaggerated_body_fails():
    kp = body_gate.normalize_keypoints(_exaggerated_coco18())
    measured = body_gate.measure_ratios(kp, HEAD_H)
    verdict, details = body_gate.assess(measured, REF, TOL)
    assert verdict == "fail"
    assert not details["hip_heads"]["ok"]
    assert not details["shoulder_hip"]["ok"]
    assert details["hip_heads"]["rel_dev"] == pytest.approx(0.444, abs=0.01)


def test_measure_partial_body_measures_what_it_can():
    kp = body_gate.normalize_keypoints(_vj_coco18())
    for missing in ("nose", "ankle_r", "ankle_l"):
        del kp[missing]
    measured = body_gate.measure_ratios(kp, HEAD_H)
    # No crown/ankles -> no height-based ratios; widths still fine.
    assert "heads_tall" not in measured
    assert "leg_fraction" not in measured
    assert measured["shoulder_hip"] == pytest.approx(1.2222, abs=0.001)
    verdict, _ = body_gate.assess(measured, REF, TOL)
    assert verdict == "pass"


def test_measure_without_head_h_skips_head_ratios():
    kp = body_gate.normalize_keypoints(_vj_coco18())
    measured = body_gate.measure_ratios(kp, None)
    assert "shoulder_heads" not in measured
    assert "heads_tall" not in measured
    assert "shoulder_hip" in measured


def test_assess_unmeasured_on_empty():
    verdict, details = body_gate.assess({}, REF, TOL)
    assert verdict == "unmeasured"
    assert details == {}


def test_assess_tolerance_boundary():
    inside = {"shoulder_hip": 1.23 * 1.14}
    verdict, details = body_gate.assess(inside, REF, TOL)
    assert verdict == "pass"
    outside = {"shoulder_hip": 1.23 * 1.16}
    verdict, details = body_gate.assess(outside, REF, TOL)
    assert verdict == "fail"
    assert details["shoulder_hip"]["rel_dev"] == pytest.approx(0.16, abs=0.001)


def test_assess_ignores_unknown_and_zero_reference():
    verdict, details = body_gate.assess(
        {"shoulder_hip": 5.0, "mystery": 1.0}, {"shoulder_hip": 0.0}, TOL)
    assert verdict == "unmeasured"


# --- decide_action -------------------------------------------------------


def test_decide_action_matrix():
    assert body_gate.decide_action("fail", "off") == "ok"
    assert body_gate.decide_action("fail", "warn") == "warn"
    assert body_gate.decide_action("fail", "reject") == "reject"
    assert body_gate.decide_action("pass", "reject") == "ok"
    assert body_gate.decide_action("unmeasured", "reject") == "ok"


def test_decide_action_unknown_mode_falls_back_to_warn():
    assert body_gate.decide_action("fail", "nonsense") == "warn"
    assert body_gate.decide_action("fail", None) == "warn"


def test_body_proportion_error_is_runtime_error():
    assert issubclass(body_gate.BodyProportionError, RuntimeError)


# --- formatting ----------------------------------------------------------


def test_format_assessment_and_failures():
    kp = body_gate.normalize_keypoints(_exaggerated_coco18())
    measured = body_gate.measure_ratios(kp, HEAD_H)
    verdict, details = body_gate.assess(measured, REF, TOL)
    assert verdict == "fail"
    summary = body_gate.format_assessment(details)
    assert "shoulder:hip" in summary and "ref 1.23" in summary
    failures = body_gate.format_failures(details, TOL)
    assert "hip width" in failures
    assert "+44" in failures  # +44.4% relative deviation
    assert "15%" in failures  # tolerance shown
