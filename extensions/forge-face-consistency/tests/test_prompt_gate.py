"""Offline tests for the prompt assembly gate (pure logic)."""

import pytest

from face_consistency import prompt_gate

_CANONICAL = (
    "21-year-old adult fictional woman, petite young woman, petite delicate frame, "
    "slim athletic build, long dark brown wavy hair with center part, dark brown eyes, "
    "tori_vj, <lora:tori_v1:0.8>, natural medium teardrop bust, never inflated, "
    "wide-set, soft shallow valley, gentle upper slope, relaxed soft shape, "
    "elegant ice-princess gown, plunging V-neckline, open-back, ultra-short hemline, "
    "sculptural cutouts, translucent skirt"
)
_CANONICAL_NEG = ("tall, curvy, statuesque, crew neck, inflated cleavage, "
                  "oversized bust, wasp waist")


def test_canonical_prompt_passes():
    r = prompt_gate.check_prompt(_CANONICAL, _CANONICAL_NEG)
    assert r["ok"] and not r["errors"] and not r["warnings"], r


def test_long_skirt_without_hem_warns():
    # 2026-10-09 failure: "long flowing skirt" dropped the ultra-short hemline.
    p = _CANONICAL.replace("ultra-short hemline", "long flowing skirt")
    r = prompt_gate.check_prompt(p, _CANONICAL_NEG)
    assert r["ok"], r  # warn-only, never blocks
    assert any("hemline" in w for w in r["warnings"]), r["warnings"]


def test_missing_identity_is_error():
    r = prompt_gate.check_prompt("a woman in a dress", "")
    assert not r["ok"] and r["errors"], r


def test_banned_words_are_error():
    r = prompt_gate.check_prompt(_CANONICAL + ", sheer fabric", _CANONICAL_NEG)
    assert not r["ok"] and any("sheer" in e for e in r["errors"]), r


def test_missing_chest_negative_warns_and_suggests():
    r = prompt_gate.check_prompt(_CANONICAL, "blurry, watermark")
    assert r["ok"], r
    assert any("anti-inflation" in w for w in r["warnings"]), r
    assert "oversized bust" in r["auto_negative"], r


def test_20262001_prompt_flags_hemline():
    # The actual 20262001 prompt: chest block present, but "long flowing
    # skirt" with no ultra-short hemline -> must warn.
    p = ("21-year-old adult fictional woman, petite young woman, petite delicate frame, "
         "slim athletic build, long dark brown wavy hair with center part, dark brown eyes, "
         "natural eyebrows, soft oval face, natural makeup, soft slight smile, tori_vj, "
         "<lora:tori_v1:0.8>, natural medium teardrop bust, never inflated, wide-set, "
         "soft shallow valley, gentle upper slope, relaxed soft shape, "
         "elegant ice-princess gown, frosty ice-blue silk, "
         "plunging V-neckline, sculptural crystal-embellished bodice, open-back, "
         "long flowing skirt with frost details, delicate silver jewelry")
    n = ("tall, curvy, statuesque, crew neck, plastic skin, inflated cleavage, "
         "wasp waist, deformed hands")
    r = prompt_gate.check_prompt(p, n)
    assert r["ok"], r
    assert any("hemline" in w for w in r["warnings"]), r["warnings"]
    assert "oversized bust" in r["auto_negative"], r


def test_retuned_chest_block_passes_without_warnings():
    # 2026-10-10 lock revision (3f18fd5): "natural medium ... never inflated"
    # is the canonical wording now; the gate must not warn on it.
    r = prompt_gate.check_prompt(_CANONICAL, _CANONICAL_NEG)
    assert r["ok"] and not r["errors"], r
    assert not any("chest block incomplete" in w for w in r["warnings"]), r["warnings"]


def test_stale_naturally_full_chest_warns():
    p = _CANONICAL.replace("natural medium teardrop bust, never inflated",
                           "naturally full teardrop bust")
    r = prompt_gate.check_prompt(p, _CANONICAL_NEG)
    assert any("chest block incomplete" in w for w in r["warnings"]), r["warnings"]


def test_cfg1_with_negative_warns_dead_negatives():
    r = prompt_gate.check_prompt(_CANONICAL, _CANONICAL_NEG, cfg_scale=1.0)
    assert r["ok"], r  # warn-only, never blocks
    assert any("CFG=1.0" in w for w in r["warnings"]), r["warnings"]


def test_cfg7_with_negative_no_cfg_warning():
    r = prompt_gate.check_prompt(_CANONICAL, _CANONICAL_NEG, cfg_scale=7.0)
    assert not any("CFG=1.0" in w for w in r["warnings"]), r["warnings"]
