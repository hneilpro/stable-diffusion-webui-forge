# TASK-001 — Forge face consistency: reference strength with max = face swap

- Status: in progress (build + repair rounds 2–3 done; critic re-review of N1/N2 and live GPU checks (F6) pending)
- Owner role: builder (Tori)
- Critic: unassigned (separate pass after build)
- Branch: feat/forge-consistent-character
- Risk: medium

## Goal
One Forge setting (enable + reference image + strength) that keeps a chosen face consistent across
SDXL and Flux generations: below max it acts as a face reference, at max it performs a full face swap.

## Non-goals
- No LoRA training, no PuLID model training/port in this task (PuLID-Flux noted as future work).
- No changes to Forge core (modules/, backend/); the feature ships as an extension under extensions/.
- No commit/push without the owner's batched-push go.

## Acceptance criteria
- [ ] Extension loads in Forge with no import errors; accordion appears in txt2img and img2img.
- [ ] With enable OFF, generation output is byte-identical to the same seed without the extension active.
- [ ] Strength < 1.0 on SDXL with a FaceID/InstantID ControlNet model present injects a ControlNet unit at weight=strength (verified in infotext/logs); without one, blended swap runs at blend=strength.
- [ ] Strength = 1.0 runs a full InsightFace swap (paste_back) + restoration on the generated image, for both SDXL and Flux checkpoints.
- [ ] Infotext records mode used (faceid/instantid/blended-swap/swap), strength, and ArcFace similarity before/after; a downgrade is never silent.
- [ ] Multi-image reference folder builds one averaged, outlier-cleaned identity template.
- [ ] Offline tests pass (import without Forge runtime, strength->mode mapping, blend math, template averaging) and are recorded in .agents/test-reports/.
- [ ] API use works via alwayson_scripts (same args as the UI).

## Required checks
- [ ] python -m py_compile on all new files; pytest for the extension's offline suite.
- [ ] Live check on the owner's 4090 via pasted tunnel URL: one SDXL gen + one Flux gen at strength 1.0, verify similarity >= 0.55 after swap, view the images. (unverified until he runs Forge)
- [ ] Critic review in .agents/reviews/ with blockers resolved or explicitly waived.

## Plan
Research: .agents/research/forge-face-consistency-research.md (done).

Blockers and caveats first:
- This VM has no GPU; live generation proof needs his 4090 + a fresh tunnel URL. Offline tests prove wiring, not pixels.
- True in-diffusion reference for Flux (PuLID-Flux) is not in Forge; Flux below max uses blended swap (honest fallback, logged).
- inswapper renders at 128px; restoration (GFPGAN if present, else detail graft) is mandatory after any swap.
- Earlier API-level A/B (2026-10-07) found ControlNet-routed FaceID accepted but pixel-inert via the external API on his build; the in-process ControlNet unit path must be re-proven live before claiming SDXL reference works.

Approach (in order):
1. `extensions/forge-face-consistency/` — install.py (insightface/onnxruntime notes), scripts/face_consistency.py
   (Script: ui() accordion, before_process() for ControlNet injection, postprocess_image() for swap/blend),
   face_consistency/ package (swap_engine ported from ~/workspace/forge-face-swap/swap.py, blend, settings),
   README.md, tests/.
2. Settings section via on_ui_settings: defaults (enable, strength, restore, verify threshold, model path).
3. Offline tests, then live test on his PC, then critic pass; loop until criteria pass.

Alternatives considered:
- Pure ReActor dependency (call the ReActor script): lost — extra install, no strength semantics, no verify gate.
- PuLID-only build: lost for now — no Flux/SDXL single backend in Forge, heavy port; swap covers both families today.

## Risks
- ControlNet unit injection shape differs between Forge versions -> mitigate by reading sd_forge_controlnet in this clone and unit-testing the args builder.
- InsightFace licensing (non-commercial research models) -> note in README, same posture as ReActor/InstantID.

## Result
Repair round 2 (2026-10-07): F1 (folder-only ControlNet unit image), F2 (extension git-ignored), F3 (shipped injection/no-op/downgrade tests), F4 (verify threshold wired to infotext/log warning), F5 (engine + template cached per run), F7 (README infotext claim scoped), F8 (blend-skip recorded) fixed; 32 offline tests pass. F6 live criteria still unverified — pending owner-4090 tunnel run (or explicit offline-only waiver). Detail: .agents/handoffs/TASK-001-repair-2-handoff.md, .agents/test-reports/TASK-001-offline-test.md.
Repair round 3 (2026-10-07): critic minors N1 (.gitignore negation re-included 13 .pyc files in git add) and N2 (representative picker tried only the first folder file) fixed; 35 offline tests pass. F6 unchanged. Detail: .agents/handoffs/TASK-001-repair-3-handoff.md.
