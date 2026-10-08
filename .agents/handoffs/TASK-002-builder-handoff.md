# Handoff — TASK-002 — builder to critic — 2026-10-08

- Task: .agents/tasks/TASK-002-api-save-and-gfpgan-docs.md
- Branch: feat/forge-consistent-character
- Worktree: ~/workspace/repos/stable-diffusion-webui-forge

## Changed files
- `modules/api/models.py` — `save_images` default `False` -> `True` for both API models (txt2img line ~108, img2img line ~129), each with a short comment. This is the whole behaviour change: `api.py` already derives `do_not_save_samples`/`do_not_save_grid` from it, so API runs now save like UI runs; explicit `"save_images": false` still opts out.
- `extensions/forge-face-consistency/README.md` — Requirements: GFPGAN bullet expanded; new "GFPGAN — two different files, two different jobs" section with exact download URLs + destination folders for the extension ONNX and the Forge-native .pth, plus the infotext confirmation signal.
- `.agents/research/2026-10-08-gfpgan-and-api-save.md`, `.agents/tasks/TASK-002-api-save-and-gfpgan-docs.md`, `.agents/test-reports/TASK-002-offline-test.md`, `.agents/BOARD.md`, `.agents/status.json` — coordination records.
- NOT touched (pre-existing in worktree, TASK-001): `scripts/forge_face_consistency.py` repair round 6, `tests/test_script_shell.py`, `.agents/test-reports/TASK-001-live-test.md`.

## Verification
- `py_compile` on touched Python files — exit 0 (test report).
- AST default check — `save_images` defaults `[True, True]`; api.py opt-out mapping intact.
- `work/test-env/venv/bin/python -m pytest extensions/forge-face-consistency/tests/ -q` — 45 passed.
- Full evidence: .agents/test-reports/TASK-002-offline-test.md

## Unverified
- Live API write + both GFPGAN paths on the owner's 4090 — unverified — no GPU/tunnel in this VM; exact checks listed in the test report. Do not call TASK-002 done until the owner runs them or waives.

## Risks / open decisions
- API-heavy workflows now write every image to disk (same folders/naming as UI). Opt-out is per-request `"save_images": false`; noted in the code comment and this handoff.
- Nothing is committed or pushed — owner's batched-push rule applies. This diff sits alongside TASK-001 repair round 6; owner decides whether they ship together.

## Next action
Critic: read-only review against .agents/workflows/05-critic-review.md (scope: TASK-002 diff only; TASK-001 live F6 remains its own gate).
