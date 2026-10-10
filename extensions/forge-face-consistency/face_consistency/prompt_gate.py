"""Prompt assembly gate for VJ generations (pure logic, no Forge imports).

Two prompt failure classes from the 2026-10-09 ice-princess run:
- a dropped wardrobe token (the "ultra-short hemline" never made it into
  the prompt; the render grew a long skirt instead), and
- chest tokens present but the negative prompt missing anti-inflation terms.

check_prompt() tests a prompt against the five canonical blocks from
tools/forge/README.md section 4 (identity, LoRA, chest, wardrobe, scene).
Warn-only by design: a bad prompt can never pass silently, and a good
prompt is never blocked. The caller may append the returned
``auto_negative`` terms to the negative prompt (logged, never silent).
"""

from __future__ import annotations

# Canonical identity block markers (prompt template v1.2).
_IDENTITY_MARKERS = ("21-year-old", "petite delicate frame")

# Chest block markers (positive prompt). Retuned 2026-10-10 (lock rev
# 3f18fd5): "naturally full" overshot vs her refs; the canonical block is
# now "natural medium ..., never inflated, ... relaxed soft shape".
_CHEST_MARKERS = ("natural medium", "teardrop", "never inflated")

# Anti-inflation terms that must be in the negative prompt.
_CHEST_NEGATIVE_TERMS = ("inflated cleavage", "oversized bust")

# VJ_TORI_CHARACTER_STANDARD.md section 5 silhouette tokens; a VJ prompt
# should carry at least two of these.
_WARDROBE_TOKENS = ("plunging", "open-back", "open back", "midriff",
                    "ultra-short hemline", "ultra short hemline",
                    "cutout", "translucent", "mini skirt", "mini dress")

# Long-garment phrasing that contradicts the section 5 short-hem default.
_LONG_GARMENT = ("long flowing skirt", "long skirt", "floor-length",
                 "floor length", "full-length gown", "full length gown",
                 "flowing gown")
_SHORT_HEM = ("ultra-short hemline", "ultra short hemline", "mini skirt",
              "mini dress", "mini hem", "short hemline")

# Wording that trips image safety filters; never in a prompt.
_BANNED = ("sheer", "see-through", "see through", "transparent", "nude")


def check_prompt(prompt, negative="", cfg_scale=None):
    """Check a VJ prompt against the canonical blocks.

    Returns {"ok", "errors", "warnings", "auto_negative", "summary"}.
    ``ok`` is False only on hard errors (missing identity block, banned
    wording). Everything else warns. ``auto_negative`` lists negative
    terms the caller should append (anti-inflation terms missing while
    the chest block is in use). ``cfg_scale`` is optional; when it is
    1.0 Forge skips unconditional conditioning and the negative prompt
    is silently ignored -- the gate warns so the caller can move
    critical constraints into positive wording.
    """
    p = (prompt or "").lower()
    n = (negative or "").lower()
    errors = []
    warnings = []
    auto_negative = []

    missing_identity = [m for m in _IDENTITY_MARKERS if m not in p]
    if missing_identity:
        errors.append("identity block incomplete; missing: "
                      + ", ".join(missing_identity))
    if "tori_vj" not in p:
        warnings.append("LoRA trigger 'tori_vj' missing from prompt")
    missing_chest = [m for m in _CHEST_MARKERS if m not in p]
    if missing_chest:
        warnings.append("chest block incomplete; missing: "
                        + ", ".join(missing_chest))
    missing_neg = [t for t in _CHEST_NEGATIVE_TERMS if t not in n]
    if missing_neg:
        warnings.append("negative prompt missing anti-inflation term(s): "
                        + ", ".join(missing_neg))
        auto_negative.extend(missing_neg)
    if cfg_scale is not None and float(cfg_scale) == 1.0 and (negative or "").strip():
        warnings.append(
            "CFG=1.0: Forge skips unconditional conditioning -- negative "
            "prompts are silently IGNORED; move every critical constraint "
            "into positive wording (auto_negative terms appended above "
            "will have no effect)")
    if not any(t in p for t in ("v-neck", "v neck", "plunging")):
        warnings.append("no V-neck/plunge token in prompt (never crew necks)")
    if "crew neck" in p and "crew neck" not in n:
        warnings.append("'crew neck' in positive prompt without negative guard")
    found = [t for t in _WARDROBE_TOKENS if t in p]
    if len(found) < 2:
        warnings.append(
            f"only {len(found)} section-5 wardrobe token(s) in prompt "
            f"({', '.join(found) or 'none'}); expected at least 2")
    if (any(g in p for g in _LONG_GARMENT)
            and not any(s in p for s in _SHORT_HEM)):
        warnings.append("long-garment phrasing with no short-hem token; "
                        "the section-5 ultra-short hemline will not render "
                        "(2026-10-09 failure)")
    banned = [b for b in _BANNED if b in p]
    if banned:
        errors.append("banned wardrobe wording in prompt (trips safety "
                      "filters): " + ", ".join(banned))

    ok = not errors
    if errors:
        summary = "prompt gate: ERRORS: " + "; ".join(errors)
    elif warnings:
        summary = "prompt gate: warnings: " + "; ".join(warnings)
    else:
        summary = "prompt gate: pass"
    return {"ok": ok, "errors": errors, "warnings": warnings,
            "auto_negative": auto_negative, "summary": summary}
