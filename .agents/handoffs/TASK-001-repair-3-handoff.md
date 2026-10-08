# Handoff — TASK-001 — builder to critic — 2026-10-07 (repair round 3)

- Task: .agents/tasks/TASK-001-forge-face-consistency.md
- Branch: feat/forge-consistent-character
- Worktree: /home/hatch/workspace/repos/stable-diffusion-webui-forge
- Scope: exactly the two new minors from the repair-round-2 critic re-review (N1, N2). Nothing that already passed was rewritten. F6 (live 4090 proof) is unchanged and still open — not fixed, not waived.

## Changed files
- `.gitignore` — N1: after the forge-face-consistency negations, added `/extensions/forge-face-consistency/**/__pycache__/` and `/extensions/forge-face-consistency/**/*.pyc`, so the negation no longer re-includes bytecode. Also removed the stray `__pycache__` dirs under the extension.
- `extensions/forge-face-consistency/face_consistency/logic.py` — N2: new pure helper `list_ref_candidates(refs_dir)` returning all sorted image paths; `pick_representative_ref` now delegates to it (same first-file result as before).
- `extensions/forge-face-consistency/face_consistency/__init__.py` — export `list_ref_candidates`.
- `extensions/forge-face-consistency/scripts/forge_face_consistency.py` — N2: `representative_rgb_for_unit` now tries every folder candidate in sorted order until one decodes; the source note names the file used and any skipped unreadable files. All-corrupt folder still falls back to the single image, or returns None → the existing loud blended-swap downgrade.
- `extensions/forge-face-consistency/tests/test_script_shell.py` — 3 new tests: `test_list_ref_candidates_sorted_and_filtered`, `test_representative_skips_corrupt_first_file`, `test_representative_all_corrupt_folder_returns_none`. Pre-existing 32 tests untouched.
- `extensions/forge-face-consistency/README.md` — one sentence: ControlNet-path folder files are tried in sorted order until one decodes.

## Verification (fresh, this pass)
- `python3 -m py_compile` on all 10 extension .py files — exit 0.
- venv pytest (pytest 9.1.1, numpy 2.5.3): `35 passed, 1 warning` (32 pre-existing + 3 new), exit 0. Warning is the pre-existing `base_url` PytestConfigWarning from repo pyproject.toml.
- Import purity: logic/settings/blend/swap_engine import with no gradio/torch/modules/insightface/onnxruntime/cv2 loaded — IMPORT_PURITY_OK, exit 0.
- Disabled-no-op harness: DISABLED_NOOP_OK, exit 0.
- N1 evidence: `git add -n extensions/forge-face-consistency` lists exactly 11 files (10 .py + README), 0 `.pyc` — re-checked after pytest regenerated `__pycache__` on disk; `git check-ignore -v` shows the pycache rule matching and other extension paths still ignored; `git diff HEAD --stat` = `.gitignore` only (Forge core untouched).

## Unverified
- F6 live checks (extension load + accordion in real Forge, byte-identical OFF output in a real generation, ControlNet injection pixel effect on the owner's build, SDXL+Flux strength=1.0 ArcFace similarity ≥ 0.55, alwayson_scripts API path) — unverified — no GPU/Forge runtime or tunnel URL in this VM. TASK-001 cannot be called done until the owner-4090 tunnel run, unless the owner explicitly waives to an offline-only scope. No waiver is recorded.

## Next action
Critic re-scores N1/N2 only; then owner runs the live 4090 check for F6 (or explicitly waives to offline-only scope).
