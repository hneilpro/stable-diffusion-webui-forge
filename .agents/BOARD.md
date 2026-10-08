# BOARD — inter-agent message board

The shared, durable board for every agent working in this repo. Read the latest entries at session start before doing anything else here. Post when you: start or finish a phase, switch roles, hand off, hit a blocker, record a critic verdict, or need a human decision.

## Rules

- Append-only. Never edit or delete another agent's entry, and never rewrite your own entry after someone has replied to it; post a correction as a new entry instead.
- Newest entry at the TOP (directly under this rules block), so the current state is the first thing read.
- One entry per event. Keep it short: what happened, evidence pointer, what is needed next.
- Sign every entry with role, task id, and date (YYYY-MM-DD).
- This board is coordination, not a task tracker and not a chat log. Durable detail belongs in tasks/, handoffs/, reviews/, test-reports/, decisions/, or research/; the board entry links to it.
- No secrets, no real personal data, no raw transcripts. Distilled findings only.

## Entry format

```text
## YYYY-MM-DD — <role> — <task-id> — <event>
<2-6 lines: what changed / what was found / verdict>
Evidence: <path or command>
Next: <owner/role + exact next action, or "none">
```

---

<!-- Post new entries below this line, newest first. -->
## 2026-10-08 — builder — TASK-002 + TASK-001 repair 6 — committed + pushed on owner go
Owner approved commit + push. Shipping together: repair round 6 (tuple script_args fix + tuple test + live round 3 report; suite 45 passed) and TASK-002 (API save_images default True for txt2img/img2img + GFPGAN two-file docs in extension README; critic approve-with-notes, self-review). Live criteria remain unverified until his PC run.
Evidence: .agents/test-reports/TASK-002-offline-test.md, .agents/reviews/TASK-002-critic-review.md
Next: owner pulls + restarts Forge, places gfpgan_1.4.onnx in models/insightface/; builder live checks (API save to outputs/, injection pixel diff, `restored by: gfpgan`) via fresh tunnel URL.

## 2026-10-08 — critic — TASK-002 — verdict: approve with notes (self-review)
Offline diff inspected firsthand: two save_images default flips + README GFPGAN section; URLs/paths byte-match code constants; no secrets; 45 tests green (per test report, re-verified commands in review pass). Not done: live criteria (API PNG in outputs/, `restored by: gfpgan` infotext) unverified until owner runs Forge on his PC — same gate style as TASK-001 F6.
Evidence: .agents/reviews/TASK-002-critic-review.md, .agents/test-reports/TASK-002-offline-test.md
Next: owner pulls (when pushed on his go) + restarts Forge + drops in gfpgan_1.4.onnx; builder runs the live checks via fresh tunnel URL.

## 2026-10-08 — builder — TASK-002 — research + plan done, building on owner request
Owner asked: GFPGAN steps for Forge + make API generations save to the output folder by default. Research found two distinct GFPGAN files (extension needs `gfpgan_1.4.onnx` in `models/insightface/`; Forge-native restore needs `GFPGANv1.4.pth` in `models/GFPGAN/`) and that API saving is gated solely by the `save_images` default in `modules/api/models.py`. Plan: flip both API defaults to True (explicit false still opts out) + exact GFPGAN steps in the extension README. Owner's message is the plan OK. TASK-001 repair round 6 stays uncommitted and untouched.
Evidence: .agents/research/2026-10-08-gfpgan-and-api-save.md, .agents/tasks/TASK-002-api-save-and-gfpgan-docs.md
Next: builder builds + offline-tests; live API-write check stays unverified until owner runs Forge; critic pass before any done claim.

## 2026-10-08 — builder — TASK-001 — live round 2: SDXL swap passes (0.938); repair round 5 pushed
Round 2 on the owner's 4090 (commit 2dc41d3): SDXL max-strength swap passes in Forge — ArcFace 0.938 measured by the extension and independently (detail-graft restore, visible seam, no GFPGAN on the PC); alwayson API path passes with it. Below-max ControlNet injection still failed silently — an exception inside the then-unguarded setup block (unit build/inject) left the plan at mode=disabled with no infotext trace; repair round 5 wraps setup so failures print, land in the infotext, and downgrade loudly to blended swap (44 offline tests pass). Flux generation is blocked on the PC: Forge 500 "You do not have CLIP state dict!" — the owner's encoders (clip_l + t5xxl_fp8 in models/text_encoder/, ae in models/VAE/) exist but were not attached; retest will attach them via the forge_additional_modules option.
Evidence: .agents/test-reports/TASK-001-live-test.md (round 2), .agents/handoffs/TASK-001-repair-5-handoff.md
Next: owner pulls + restarts Forge; builder retest round 3 (injection self-report + pixel diff; Flux swap with encoders attached); critic re-scores F6 only.

## 2026-10-07 — builder — TASK-001 — repair round 4 committed + pushed on owner go
Owner approved commit + push of the API string-ref fix (script + 7 tests + live test report + handoff). Owner pulls on the PC and restarts Forge, then relays readiness (fresh tunnel URL if cloudflared restarted).
Evidence: .agents/test-reports/TASK-001-live-test.md, .agents/handoffs/TASK-001-repair-4-handoff.md
Next: owner pulls + restarts Forge; builder re-runs the live matrix (SDXL swap >= 0.55, below-max injection pixel diff, Flux swap); critic re-scores F6 only.

## 2026-10-07 — builder — TASK-001 — live round 1 done: loads + OFF identical; API string-ref bug found and fixed locally
First owner-4090 live run: extension loads (script-info, txt2img+img2img), enable OFF byte-identical to no-script (same sha256), but swap and below-max injection never ran — the API sends the reference as a base64 string and _ref_to_rgb died on shape[-1] of a 0-d array (owner's PC traceback confirms line 117; E/F pixel-identical to the OFF baseline, swap image ArcFace 0.049 = baseline). Repair round 4: ref conversion accepts base64/data-URI/path/dict and returns None instead of raising; 42 offline tests pass (7 new); the extension engine run agent-side on a live output scores 0.049 -> 0.921, so the swap core works. Not committed/pushed yet.
Evidence: .agents/test-reports/TASK-001-live-test.md, .agents/handoffs/TASK-001-repair-4-handoff.md
Next: owner go to commit+push, then owner pulls + restarts Forge; builder re-runs the live matrix (SDXL swap, injection pixel diff, Flux swap); critic re-scores F6 only.

## 2026-10-07 — builder — TASK-001 — repair round 3 done, ready for critic re-review
Critic minors fixed: N1 .gitignore now ignores the extension's __pycache__/*.pyc after the negation (git add -n = exactly 11 real files, 0 bytecode, re-checked after pytest regenerated caches; core diff = .gitignore only); N2 representative picker now tries every folder file in sorted order until one decodes (new logic.list_ref_candidates; skipped files named in the source note; all-corrupt folder keeps the loud downgrade). Fresh evidence: py_compile 10/10 exit 0, pytest 35 passed (32 pre-existing + 3 new), IMPORT_PURITY_OK, DISABLED_NOOP_OK. F6 live 4090 proof still unverified and unwaived — task is not done.
Evidence: .agents/handoffs/TASK-001-repair-3-handoff.md, .agents/test-reports/TASK-001-offline-test.md (repair round 3 addendum)
Next: critic re-scores N1/N2 only; owner runs the live 4090 check for F6 (or explicitly waives to offline-only scope).

## 2026-10-07 — critic — TASK-001 — re-review verdict: needs changes (F6 only)
Repair round 2 verified firsthand: F1 fixed (real decoded folder representative for the ControlNet unit; empty folder → loud downgrade — reproduced with a real PIL image), F2 fixed (.gitignore negations; git status/add -n list the extension, core diff = .gitignore only), F3/F4/F5/F7/F8 fixed; 32 offline tests + py_compile + import purity re-run green in this pass. New minors: N1 git add -n also lists 13 __pycache__ .pyc files (negation overrides the __pycache__ ignore — fix before commit), N2 representative picker tries only the first folder file. Gate blocker remains F6: live 4090 proof (load, byte-identical OFF, injection pixel effect, SDXL+Flux similarity ≥ 0.55, API) still unverified and unwaived — task is not done.
Evidence: .agents/reviews/TASK-001-critic-review.md (re-review section)
Next: owner runs the live 4090 check for F6 (or explicitly waives to offline-only scope); builder fixes N1 before the batched commit.

## 2026-10-07 — builder — TASK-001 — repair round 2 done, ready for critic re-review
F1 folder-only ControlNet unit now uses a decoded folder representative (empty folder → loud blended-swap downgrade); F2 .gitignore fixed (`/extensions/*` + negations — `git status`/`git add -n` list the extension, only tracked change is .gitignore); F3 shipped tests/test_script_shell.py (18 tests: no-op, injection slots, F1, downgrade params, verify warning, blend-skip); F4 verify threshold now warns in infotext/log below threshold; F5 engine cached on p + template cached per ref source; F7 README claim scoped; F8 blend-skip recorded. 32 offline tests pass, py_compile clean, import purity re-verified. F6 live checks still unverified (no GPU/tunnel here).
Evidence: .agents/handoffs/TASK-001-repair-2-handoff.md, .agents/test-reports/TASK-001-offline-test.md (repair addendum)
Next: critic re-scores F1–F5/F7/F8; owner-4090 live run for F6 (or explicit offline-only waiver).

## 2026-10-07 — critic — TASK-001 — verdict: needs changes
Independent critic pass done. Offline core is sound (no-op, strength semantics, Flux guard, template, restore, 14 tests re-run green), but 3 blockers: F1 folder-only reference breaks SDXL ControlNet path (unit image = np.asarray(None)); F2 extension is git-ignored (.gitignore /extensions) so a push would omit it; F6 live checks still unverified. Majors: shipped tests miss injection/no-op coverage, verify-threshold setting is dead, engine rebuilt per image.
Evidence: .agents/reviews/TASK-001-critic-review.md
Next: builder fixes F1/F2 (+F3-F5), then owner-4090 live run for F6; critic re-scores failed rows only.

## 2026-10-07 — builder — TASK-001 — build done, offline tests green
Extension built under extensions/forge-face-consistency/ (Forge core untouched): AlwaysVisible script with Enable + reference image + optional refs folder + Strength slider (default 0.85); strength=max = full swap+restore in postprocess_image; strength<max on SDXL injects a ControlNet FaceID/InstantID unit at weight=strength (first free slot of p.script_args, matching lib_controlnet.external_code.ControlNetUnit), else blended swap; Flux never gets SDXL adapters. Mode/strength/similarity before-after written to infotext; downgrades logged, never silent. Global defaults via on_ui_settings; API via alwayson_scripts. Swap engine ported from the proven swap.py (buffalo_l, averaged outlier-cleaned template, ArcFace verify, GFPGAN-or-detail-graft restore).
Evidence: python -m py_compile OK on all new files; pytest 14 passed (offline, bare venv: strength->mode mapping, family detection, adapter picking, blend math, template averaging). Script imports cleanly without Forge runtime.
Next: critic review; live SDXL+Flux proof still needs owner 4090 + tunnel URL (unverified).

## 2026-10-07 — builder — TASK-001 — plan posted, awaiting owner OK
Cloned feat/forge-consistent-character (empty branch at main b9a772f). Research done: Forge natively supports IP-Adapter FaceID/InstantID via ControlNet; swap (inswapper) is the model-agnostic route for SDXL+Flux. Plan: extension under extensions/forge-face-consistency, strength<max = reference, max = full swap + restore, ArcFace verify in infotext.
Evidence: .agents/research/forge-face-consistency-research.md, .agents/tasks/TASK-001-forge-face-consistency.md
Next: owner OK on plan, then Build phase.
