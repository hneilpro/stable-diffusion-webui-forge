# Test report — TASK-001 — 2026-10-07

- Branch: feat/forge-consistent-character
- Changed behavior: extensions/forge-face-consistency offline logic — pure-logic imports without Forge/gradio/torch, strength→mode mapping, blend math, template averaging, and enable-OFF postprocess no-op.

## Commands run

- `python3 -m py_compile extensions/forge-face-consistency/face_consistency/__init__.py extensions/forge-face-consistency/face_consistency/blend.py extensions/forge-face-consistency/face_consistency/logic.py extensions/forge-face-consistency/face_consistency/settings.py extensions/forge-face-consistency/face_consistency/swap_engine.py extensions/forge-face-consistency/install.py extensions/forge-face-consistency/scripts/forge_face_consistency.py extensions/forge-face-consistency/tests/conftest.py extensions/forge-face-consistency/tests/test_logic.py` — all 9 files compile — exit 0
- `python3 -m pytest extensions/forge-face-consistency/tests -q` (system python3 3.12.3) — could not run: `No module named pytest` — exit 1. Not counted as a pass; superseded by the venv run below.
- `work/test-env/venv/bin/python -m pytest extensions/forge-face-consistency/tests -q` (isolated venv created in the workflow work dir, Python 3.12.3, pytest 9.1.1, numpy 2.5.3; system python3 has numpy 1.26.4 but no pytest) — 14 passed, 1 warning (PytestConfigWarning: unknown config option `base_url` from the repo pyproject.toml — pre-existing, unrelated to this extension) — exit 0
- `python3 -m unittest discover -s extensions/forge-face-consistency/tests -v` — FAILED (errors=1), exit 1: test_logic.py imports pytest, which system python3 lacks. The suite is pytest-style (plain functions + `pytest.approx`), so unittest discovery is not a valid runner for it. Recorded here so the fallback path is not mistaken for a pass.
- Import-purity check (system python3, `sys.path` = extensions/forge-face-consistency): `import face_consistency.logic / settings / blend / swap_engine` succeeded; `gradio`, `torch`, `modules`, `insightface`, `onnxruntime` absent from `sys.modules` after import, and `cv2` also not loaded at import time (lazy inside functions) — exit 0
- Offline script-shell import: importing `scripts/forge_face_consistency.py` without Forge stubs succeeds with `_FORGE_AVAILABLE=False` and does not load gradio/torch — exit 0
- Disabled-noop harness (work dir `check_disabled_noop.py`, stubs gradio + modules.* so `_FORGE_AVAILABLE=True`, then calls the real `FaceConsistencyScript` methods): `postprocess_image` with plan.enabled=False returns None and leaves `pp.image` as the identical sentinel object; same with no plan at all; `before_process` with enabled=False leaves mode `disabled` and writes no extra_generation_params; `postprocess` with enabled=False leaves infotexts unchanged — `DISABLED_NOOP_OK` — exit 0

## Live / manual checks

- Strength→mode mapping exercised directly via `logic.decide_mode` for strength 0, 0.5, 0.85, 1.0 × family sdxl/flux/other × adapter present/absent — exit 0. Observed: 0 → `disabled` in all cases; 1.0 → `swap` (not downgraded) for every family; mid (0.5/0.85) → `controlnet` only for sdxl+adapter, otherwise `blended-swap` with downgraded=True and a reason naming the cause (Flux, missing adapter, or unsupported family).

## Outputs inspected

- pytest verbose listing: 14 tests in tests/test_logic.py all passed — max-strength swap for every family, SDXL controlnet vs loud downgrade, Flux never offered SDXL adapter, zero-strength disabled, clamp/is_max helpers, family detection (incl. Flux winning over a stale sdxl flag), FaceID-preferred adapter picking, blend endpoints/midpoint/mask/clamp, feather-mask soft edges, template outlier drop, template single/identical.
- No image outputs in scope for offline tests; nothing visual claimed.

## Failures

- System-python pytest run: `No module named pytest` — environment gap, not caused by this change; resolved by running the suite under an isolated venv with pytest 9.1.1 (documented above).
- unittest discovery failure — same root cause (pytest absent for system python3) plus the suite being pytest-style; not a code failure. Classified: pre-existing/environment, with evidence above.

## Skipped / unverified

- Live Forge load (accordion in txt2img/img2img, real ControlNet injection, real swap on SDXL + Flux, similarity ≥ 0.55 after swap, byte-identical output with enable OFF in a real generation) — unverified — this VM has no GPU/Forge runtime or tunnel URL in this run; verify on the owner's 4090 via a pasted trycloudflare URL per TASK-001 required checks.
- `install.py` execution (would pip-install insightface/onnxruntime) — unverified — intentionally not run offline; it was only py_compile-checked and AST-parsed. Verify at first real Forge enable.
- API path via alwayson_scripts with the UI arg order — unverified — needs the live Forge check above.

## Repair round 2 addendum — 2026-10-07

Critic findings F1–F5, F7, F8 addressed (F6 live checks remain unverified, see below).

- `python3 -m py_compile` on all 10 extension .py files (incl. new tests/test_script_shell.py) — exit 0
- venv pytest (pytest 9.1.1, numpy 2.5.3): `32 passed` (14 pre-existing + 18 new), exit 0. New tests/test_script_shell.py covers: disabled no-op (postprocess_image/before_process/postprocess leave image, params, infotexts untouched); ControlNet slot injection (first disabled slot wins, dict units, all-busy → False, no ControlNet script → False); F1 folder-only reference builds a unit with a real decoded ndarray (never np.asarray(None)) and an empty folder downgrades loudly with the reason in extra_generation_params; Flux downgrade recorded in params; F4 below-threshold similarity writes `FaceConsistency verify` + plan flag, good similarity does not; F8 blend-skip recorded in params + plan; representative picking (sorted first image, folder overrides single image); is_below_threshold; template cache keys separate distinct sources.
- Import purity re-run after changes: logic/settings/blend/swap_engine import with no gradio/torch/modules/insightface/onnxruntime/cv2 loaded — exit 0.
- Engine cache: two `_get_engine(p)` calls on the same p return the identical instance (ENGINE_CACHE_OK); FaceSwapEngine also caches the built template per ref_source.
- Git visibility (F2): `.gitignore` now `/extensions/*` + `!/extensions/forge-face-consistency/` + `!/extensions/forge-face-consistency/**`; `git status --porcelain` lists `?? extensions/forge-face-consistency/`, `git add -n` lists all 10 files, other extension paths still ignored, `git diff HEAD --stat` shows only `.gitignore` (Forge core untouched).

## Coverage gaps

- swap_engine's InsightFace-dependent paths (build_template on real photos, swap + GFPGAN/detail-graft restore, similarity on real faces) have no offline coverage beyond the pure `template_embedding`/`cosine` helpers — acceptable for this offline task; they are covered by the pending live check, not by this report.
- ControlNet unit injection (`_inject_controlnet_unit` slot rewriting) is exercised only by code reading plus the disabled-path harness; the enabled injection path remains unverified until the live Forge run.

## Re-run after repair round 2 — verification — 2026-10-08

Fresh re-run of the same offline checks (py_compile + full test suite + enable-OFF no-op) after repair round 2, on branch `feat/forge-consistent-character`. All commands run from `/home/hatch/workspace/repos/stable-diffusion-webui-forge`.

- `python3 -m py_compile` on all 10 extension .py files (`face_consistency/__init__.py`, `blend.py`, `logic.py`, `settings.py`, `swap_engine.py`, `install.py`, `scripts/forge_face_consistency.py`, `tests/conftest.py`, `tests/test_logic.py`, `tests/test_script_shell.py`) — exit 0 (10 files).
- `work/test-env/venv/bin/python -m pytest extensions/forge-face-consistency/tests -q` (Python 3.12.3, pytest 9.1.1, numpy 2.5.3) — `32 passed, 1 warning in 0.62s` — exit 0. Warning is the pre-existing `PytestConfigWarning: Unknown config option: base_url` from repo `pyproject.toml`, unrelated to this extension.
- Verbose listing (`-v`) confirms all 32: 14 in `test_logic.py` (max-strength swap, SDXL controlnet vs downgrade, Flux adapter exclusion, zero-strength disabled, helpers, family detection, adapter picking, blend endpoints/mask/clamp, feather mask, template outlier/single) + 18 in `test_script_shell.py` (disabled no-op ×4, ControlNet injection ×4, folder-only F1 ×2, representative picking, Flux downgrade recorded, blend-skip F8, verify-threshold F4 ×2, pick_representative, is_below_threshold, template cache keys) — exit 0.
- Disabled-noop harness (`work/check_disabled_noop.py`, stubs gradio + modules.* so `_FORGE_AVAILABLE=True`): `postprocess_image` disabled returns None and leaves `pp.image` as identical sentinel; no-plan case same; `before_process` disabled leaves mode `disabled` and `extra_generation_params == {}`; `postprocess` disabled leaves infotexts `['hello']` unchanged — `DISABLED_NOOP_OK` — exit 0.
- Import-purity re-check (system python3): `face_consistency.logic/settings/blend/swap_engine` import with `gradio`/`torch`/`modules`/`insightface`/`onnxruntime`/`cv2` absent from `sys.modules` — `IMPORT_PURITY_OK` — exit 0.
- Offline script-shell import (no Forge stubs): `_FORGE_AVAILABLE=False`, `SCRIPT_SHELL_IMPORT_OK` — exit 0.

Result: all offline checks pass on fresh execution. Live Forge checks (GPU/4090, real swap similarity, ControlNet injection in a live generation) remain unverified — not in scope for this offline re-run.

## Repair round 3 addendum — 2026-10-07

Scope: critic minors N1 (bytecode re-included by the .gitignore negation) and N2 (representative picker tried only the first folder file). F1–F5/F7/F8 code untouched.

- `.gitignore`: added `/extensions/forge-face-consistency/**/__pycache__/` and `/extensions/forge-face-consistency/**/*.pyc` after the negations; removed stray `__pycache__` dirs. `git add -n extensions/forge-face-consistency` now lists exactly 11 files (10 .py + README), 0 `.pyc` — verified again after a pytest run regenerated the caches on disk. `git check-ignore -v` confirms the pycache rule matches and `extensions/some-other-ext/foo.py` is still ignored. `git diff HEAD --stat` = `.gitignore` only.
- N2: `logic.list_ref_candidates` (new, pure) returns all sorted image paths; `representative_rgb_for_unit` tries each until one decodes and records skipped files in the source note. All-corrupt folder → single-image fallback or None → existing loud downgrade.
- `python3 -m py_compile` on all 10 extension .py files — exit 0.
- venv pytest (pytest 9.1.1, numpy 2.5.3): `35 passed, 1 warning` — exit 0 (32 pre-existing + 3 new N2 tests: candidates listing sorted/filtered, corrupt-first-file skipped, all-corrupt folder returns None). Warning is the pre-existing `base_url` PytestConfigWarning.
- Import purity re-run: IMPORT_PURITY_OK — exit 0. Disabled-no-op harness: DISABLED_NOOP_OK — exit 0.
- Live Forge checks remain unverified (F6) — no GPU/Forge runtime in this VM; pending the owner-4090 tunnel run or an explicit offline-only waiver.

## Re-run after repair round 3 — verification — 2026-10-07

Fresh independent re-run of the same offline checks (py_compile + full test suite + enable-OFF no-op) after repair round 3, on branch `feat/forge-consistent-character` (HEAD `b9a772f`). All commands run from `/home/hatch/workspace/repos/stable-diffusion-webui-forge`.

- `python3 -m py_compile` on all 10 extension .py files (`face_consistency/__init__.py`, `blend.py`, `logic.py`, `settings.py`, `swap_engine.py`, `install.py`, `scripts/forge_face_consistency.py`, `tests/conftest.py`, `tests/test_logic.py`, `tests/test_script_shell.py`) — exit 0 (10 files).
- `work/test-env/venv/bin/python -m pytest extensions/forge-face-consistency/tests -q` (Python 3.12.3, pytest 9.1.1, numpy 2.5.3) — `35 passed, 1 warning in 0.67s` — exit 0. Warning is the pre-existing `PytestConfigWarning: Unknown config option: base_url` from repo `pyproject.toml`, unrelated to this extension.
- Verbose listing (`-v`) confirms all 35: 14 in `test_logic.py` (max-strength swap, SDXL controlnet vs downgrade, Flux adapter exclusion, zero-strength disabled, helpers, family detection, adapter picking, blend endpoints/mask/clamp, feather mask, template outlier/single) + 21 in `test_script_shell.py` (disabled no-op x4, ControlNet injection x4, folder-only F1 x2, representative picking, Flux downgrade recorded, blend-skip F8, verify-threshold F4 x2, pick_representative, is_below_threshold, N2 candidates sorted/filtered, N2 corrupt-first-file skipped, N2 all-corrupt returns None, template cache keys) — exit 0.
- Disabled-noop harness (`work/check_disabled_noop.py`, stubs gradio + modules.* so `_FORGE_AVAILABLE=True`): `postprocess_image` disabled returns None and leaves `pp.image` as identical sentinel; no-plan case same; `before_process` disabled leaves mode `disabled` and `extra_generation_params == {}`; `postprocess` disabled leaves infotexts `['hello']` unchanged — `DISABLED_NOOP_OK` — exit 0.
- Import-purity re-check (system python3): `face_consistency.logic/settings/blend/swap_engine` import with `gradio`/`torch`/`modules`/`insightface`/`onnxruntime`/`cv2` absent from `sys.modules` — `IMPORT_PURITY_OK` — exit 0.
- Offline script-shell import (no Forge stubs): `_FORGE_AVAILABLE=False`, `SCRIPT_SHELL_IMPORT_OK` — exit 0.
- Git visibility / N1 re-check: `git add -n extensions/forge-face-consistency` lists exactly 11 files (10 .py + README), 0 `.pyc`; `git check-ignore -v` confirms pycache rule matches (`:46:/extensions/forge-face-consistency/**/__pycache__/`) and `extensions/some-other/foo.py` still ignored (`.gitignore:43:/extensions/*`); `git diff HEAD --stat` = `.gitignore` only.

Result: all offline checks pass on fresh execution after repair round 3. Live Forge checks (GPU/4090, real swap similarity, ControlNet injection in a live generation) remain unverified — not in scope for this offline re-run.
