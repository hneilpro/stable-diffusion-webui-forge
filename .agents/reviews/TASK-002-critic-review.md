# Critic review — TASK-002 — 2026-10-08

- Reviewer: Tori (self-review — not independent; same agent built and graded in separate passes. Downgrade confidence accordingly.)
- Scope graded: TASK-002 diff only (`modules/api/models.py`, extension `README.md`, task/research/test-report/handoff artifacts). TASK-001 repair round 6 and F6 were not re-graded here.
- Verdict: **approve with notes** — offline scope passes; the task is NOT done until the live criteria run on the owner's PC (see Blocking-for-done note).

## Rubric scores (inspected firsthand in this pass)
1. Research: PASS — questions were decision-shaped; primary sources are this repo's own code (`gfpgan_model.py:18-19` model_url, `cmd_args.py:55` default `models/GFPGAN`, `swap_engine.py:180-184` ONNX search paths) plus one web check for the ONNX distribution point; uncertainty (owner-PC behaviour) named.
2. Plan/Design: PASS — acceptance criteria checkable; rejected alternatives (force-save in api.py, new shared option) are real trade-offs, not theatre; owner's request message serves as the plan OK for this small change. No ADR: the change is a reversible one-line-per-endpoint default, not hard to reverse (flip back), so an ADR would be ceremony.
3. Build: PASS — diff is exactly two default flips + comments in `modules/api/models.py` and docs in the extension README; style matches surrounding field dicts; no new dependencies; secrets scan of the full diff: none found.
4. Test: PASS (offline) — py_compile exit 0; AST check shows both defaults `[True, True]` and the api.py opt-out mapping unchanged; extension suite 45 passed. The check-script's own first-run bug was reported in the test report instead of buried — good. No HTTP-level test exists upstream; correctly flagged as a coverage gap rather than silently passed.
5. Correctness: PASS on source evidence — save path traced `models.py` default -> `api.py` `do_not_save_* = not save_images` -> `processing.py:513-515` `save_samples()` -> `outdir_*_samples`. Grids follow the same flag (disclosed in the task plan). README URLs/paths byte-match the code constants (verified by grep in this pass).
6. Evidence quality: PASS — a stranger can reproduce every offline claim from the test report's commands.
7. Risk: PASS with notes — disk growth on API-heavy runs is real and disclosed, with a working per-request opt-out. Rollback is trivial (revert two lines / don't pull).
8. Suggestions: none load-bearing. Opinion only: if the owner later wants the Extras GFPGAN panel to also default-on, that's a separate `gfpgan_visibility` default decision — not part of this task.

## Blocking-for-done note (not a code defect)
- The task's live acceptance criteria (API call writes a PNG to `outputs/txt2img-images/`; swap infotext reads `restored by: gfpgan`; native GFPGAN restore runs) are `unverified — no GPU/tunnel in this VM`, exactly as recorded. Any "TASK-002 done" claim before the owner's PC run — or his explicit waiver — would be an overclaim. TASK-001's F6 gate is unchanged by this review.

## Findings
- Blocking: none in the offline diff.
- Major: none.
- Minor: the `TASK-002` tag in the two code comments ties core code to an internal task id; harmless and useful for future blame, noted for consistency only.
