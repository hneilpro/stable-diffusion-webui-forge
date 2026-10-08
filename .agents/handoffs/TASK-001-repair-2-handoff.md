# Handoff — TASK-001 — builder to critic — 2026-10-07

- Task: .agents/tasks/TASK-001-forge-face-consistency.md
- Branch: feat/forge-consistent-character
- Worktree: /home/hatch/workspace/repos/stable-diffusion-webui-forge

## Changed files
- .gitignore — `/extensions` → `/extensions/*` + negations for forge-face-consistency (F2: extension was invisible to git; `git add -n` now lists all files, other extensions stay ignored). Only tracked change.
- extensions/forge-face-consistency/scripts/forge_face_consistency.py — F1: ControlNet unit image now comes from `representative_rgb_for_unit` (folder representative decoded, folder overrides single image); no readable image → loud downgrade to blended-swap. F4: verify threshold consumed — below-threshold similarity writes `FaceConsistency verify` param + log warning + infotext note; ControlNet path records similarity as `n/a (controlnet)`. F5: engine cached on `p` via `_get_engine`, cleared per generation. F8: blend-skip (no face in swapped image) logged + recorded in params/infotext.
- extensions/forge-face-consistency/face_consistency/logic.py — added `VERIFY_THRESHOLD`, `is_below_threshold`, `pick_representative_ref`, `IMAGE_EXTENSIONS` (pure, offline-testable).
- extensions/forge-face-consistency/face_consistency/swap_engine.py — template cache per ref_source (`_template_key`) so batches build the averaged identity once (F5).
- extensions/forge-face-consistency/face_consistency/__init__.py — export the new helpers.
- extensions/forge-face-consistency/README.md — infotext claim scoped: similarity measured on swap paths, `n/a (controlnet)` on reference paths; verify warning documented (F7).
- extensions/forge-face-consistency/tests/test_script_shell.py — NEW, 18 tests: disabled no-op (ported harness), ControlNet slot injection (first-disabled wins, dict units, all-busy, no-ControlNet), F1 folder-only unit image + empty-folder downgrade, Flux downgrade recorded in params, F4 verify warning on/off, F8 blend-skip recorded, representative picking, threshold helper, template cache keys (F3).

## Verification
- `python3 -m py_compile` on all 10 extension .py files — exit 0 (full evidence: .agents/test-reports/TASK-001-offline-test.md, repair round 2 addendum)
- venv pytest (`pytest 9.1.1`, `numpy 2.5.3`) — 32 passed (14 pre-existing + 18 new), exit 0
- Import purity re-checked: no gradio/torch/modules/insightface/onnxruntime/cv2 after importing logic/settings/blend/swap_engine — exit 0
- Engine cache identity: two `_get_engine(p)` calls return the same instance — OK
- Git: `git status --porcelain` shows `?? extensions/forge-face-consistency/`; `git diff HEAD --stat` = only `.gitignore` (Forge core untouched)

## Unverified
- Live Forge checks (extension load + accordion, byte-identical OFF output, ControlNet injection pixel effect, SDXL+Flux strength=1.0 similarity ≥ 0.55, alwayson_scripts API) — unverified — no GPU/Forge runtime or tunnel URL in this VM; needs the owner-4090 tunnel run (F6, pending, not failed).
- `install.py` execution — unverified — py_compile only; verify at first real Forge enable.

## Risks / open decisions
- F6 remains the gate blocker: TASK-001 cannot be called done until the live run, unless the owner explicitly waives to an offline-only scope.

## Next action
Critic re-scores the previously failed rows (F1–F5, F7, F8) on this diff; then owner runs the live 4090 check for F6.
