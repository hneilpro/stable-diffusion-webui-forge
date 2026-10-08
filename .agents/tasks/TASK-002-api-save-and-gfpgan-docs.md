# TASK-002 — API generations save to the output folder by default + GFPGAN setup docs

- Status: in progress
- Owner role: builder (Tori)
- Critic: Tori (separate critic pass, read-only)
- Branch: feat/forge-consistent-character
- Risk: low

## Goal
API (txt2img/img2img) generations save their output images to the normal output folders by default — like UI runs — while a caller can still opt out per request; and the owner gets exact, no-homework steps for the two GFPGAN files Forge can use.

## Non-goals
- No change to the UI generation path (already governed by `opts.samples_save`, default True).
- No new settings option, no forced save that ignores an explicit `"save_images": false`.
- No GFPGAN weights committed to the repo; the extension keeps discovering `gfpgan_1.4.onnx` where it already looks.
- TASK-001's uncommitted repair round 6 (script_args tuple fix) is untouched and not part of this diff's claim.

## Acceptance criteria
- [ ] `modules/api/models.py`: `save_images` default is `True` for both `StableDiffusionTxt2ImgProcessingAPI` and `StableDiffusionImg2ImgProcessingAPI`; explicit `false` in a request still yields `do_not_save_samples=True` (mapping in `modules/api/api.py` unchanged).
- [ ] Extension README Requirements section names both GFPGAN files, their exact download URLs, and their exact destination folders (extension ONNX vs Forge-native .pth), plus the infotext confirmation signal.
- [ ] Offline checks pass: `py_compile` on touched files, AST-level default check, existing extension pytest suite green.
- [ ] Critic review recorded; live API write check on the owner's PC marked `unverified` until he runs Forge (no GPU/tunnel here).

## Required checks
- [ ] `python3 -m py_compile modules/api/models.py modules/api/api.py`
- [ ] AST/scripted check that both API models declare `save_images` default `True`
- [ ] `python -m pytest extensions/forge-face-consistency/tests/ -q` (regression: repair-round-6 work still green)
- [ ] Live check on owner's 4090: one API txt2img call without `save_images`, confirm the PNG lands in `outputs/txt2img-images/` (unverified until he runs Forge)

## Plan
Research: .agents/research/2026-10-08-gfpgan-and-api-save.md (done).

Blockers and caveats first:
- This VM has no GPU and no tunnel to his PC; the live criterion stays `unverified` here, same as TASK-001's F6.
- The worktree already carries TASK-001 repair round 6 uncommitted changes (`forge_face_consistency.py`, `test_script_shell.py`, live-test report). This task must not stage, revert, or claim them.

Approach:
1. `modules/api/models.py` — flip `save_images` default `False` -> `True` in the two `additional_fields` entries (single source of the API default; `api.py` already derives `do_not_save_*` from it).
2. `extensions/forge-face-consistency/README.md` — expand the Requirements bullet into exact GFPGAN steps for both models.
3. Test offline (above), write test report + handoff, update board/status, critic pass.

Alternatives considered:
- Force `do_not_save_samples=False` in `api.py`: rejected — silently ignores a caller's explicit opt-out.
- New shared option "save API images by default": rejected — more surface (settings UI, persistence) for a behaviour the owner asked to be the plain default; callers keep the per-request flag.

## Risks
- Disk use on API-heavy runs grows (every API image now writes to disk). Mitigation: documented opt-out (`"save_images": false`) and same output rotation/naming as UI; flagged in README/handoff.
- Importers that parse the OpenAPI schema may notice the default change. Acceptable: it is the requested behaviour, and the field itself is unchanged.

## Result
Built 2026-10-08; committed + pushed same day on owner's go (with TASK-001 repair round 6): `modules/api/models.py` save_images defaults flipped to True for both endpoints; extension README GFPGAN section added. Offline evidence: .agents/test-reports/TASK-002-offline-test.md (py_compile OK, AST defaults [True, True], 45 extension tests passed). Critic: .agents/reviews/TASK-002-critic-review.md — approve with notes (self-review). Live criteria (API PNG lands in outputs/, GFPGAN infotext signals) remain `unverified — owner's PC run` — task is not done until those pass or he waives.
