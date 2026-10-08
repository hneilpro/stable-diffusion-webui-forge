# Critic review — TASK-001 — 2026-10-07

- Reviewer: critic
- Reviewed: branch feat/forge-consistent-character @ b9a772f, extensions/forge-face-consistency/ (10 files), .agents/tasks/TASK-001-forge-face-consistency.md, .agents/test-reports/TASK-001-offline-test.md, .agents/research/forge-face-consistency-research.md
- Independence: independent pass (critic subagent; did not write the extension, tests, or test report). All findings below were re-verified firsthand in this pass: full source read, py_compile re-run, pytest re-run (14 passed), disabled-no-op harness re-run (DISABLED_NOOP_OK), ControlNet/Forge internals read in this clone, injection-slot logic exercised with stubs.

## Verdict
needs changes

## Rubric scores

### Process
1. Research: PASS — .agents/research/forge-face-consistency-research.md verified against this clone firsthand: preprocessor names in logic.py match extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py:100-107 exactly; ControlNetUnit kwargs used by _build_controlnet_unit exist in lib_controlnet/external_code.py:157-182; prior live A/B caveat (API FaceID pixel-inert) is carried into the plan as a re-proof obligation.
2. Plan/Design: PASS — checkable acceptance criteria, blockers-first, real alternatives weighed. Note: TASK header still says "plan gate — awaiting owner OK" while the build is done; owner-OK evidence not in repo (Unverified, minor).
3. Build: PASS with findings — diff is scoped to the extension (tracked Forge core untouched: git diff HEAD empty); dependencies spelled out (install.py, README); no secrets or real names anywhere (grep-verified). Findings F2/F4/F5 below keep this from being clean.
4. Test: FAIL — shipped suite passes but does not cover several claims it is cited for; see F3.

### Outcome
5. Correctness: FAIL — strength semantics (max=swap, below=reference/blended), family guard (Flux never receives SDXL adapters), multi-ref outlier-cleaned template, restore-after-swap, downgrade logging/infotext all verified in code, but two load-bearing defects fail: F1 (folder-only reference breaks the SDXL ControlNet path) and F2 (deliverable invisible to git). Live criteria remain unverified (F6) — honestly labelled, but not met.
6. Evidence quality: PASS — test report gives exact commands, exit codes, and honest skips (system-python pytest failure and unittest failure both recorded, not hidden). Weakened by F3: the disabled-no-op harness lives only in the workflow work dir, not in the repo, so a stranger cannot re-run it from the repo alone.
7. Risk: FAIL — the git-invisibility risk (F2) is undisclosed anywhere in the task/board/report and would silently swallow the whole feature at push time. Other top risks (injection pixel-effect, CPU-provider swap cost) are disclosed.
8. Suggestions: PASS — suggestions below cite measured/repo backing or are labelled opinion.

## Findings

- [Blocking] F1 — Folder-only reference breaks the SDXL below-max ControlNet path. UI/README advertise the reference folder as overriding the single image (script ui() refs_dir, README "Use" step 2). In before_process (scripts/forge_face_consistency.py:217-221) the ControlNet unit is built from ref_image even when only refs_dir was given: _build_controlnet_unit receives None, so unit.image = np.asarray(None) — a 0-d object array (reproduced with stubs in this pass). Forge's get_input_data (extensions-builtin/sd_forge_controlnet/scripts/controlnet.py, SIMPLE branch) expects an HWC ndarray or None (None falls back to the a1111 init image): the 0-d object array will error or condition on the wrong image, while our infotext still records mode=faceid/instantid. That is a wrong-result-with-confident-infotext failure on an advertised input combination. Fix: when only a folder is given, load a representative reference image (first detectable face file) for the unit image, or downgrade loudly to blended-swap (whose template path already handles folders) and record the downgrade.
- [Blocking] F2 — The extension is invisible to git, so it cannot ship as claimed and the "core untouched" evidence is vacuous. Repo .gitignore:43 is `/extensions`; `git check-ignore -v` confirms extensions/forge-face-consistency/** is ignored, `git status --porcelain` shows nothing for the extension, and `git ls-files extensions` tracks only "put extensions here.txt". The owner's batched `git add`/push flow would silently omit the entire feature. Fix: add a negation (e.g. `!/extensions/forge-face-consistency/` plus `!/extensions/forge-face-consistency/**`) or relocate to extensions-builtin/, then show `git status` listing the files. Until then "Forge core untouched (git status)" proves nothing about the new code, and the branch does not contain the deliverable.
- [Blocking] F6 — Live acceptance criteria are unmet (pending, not failed): extension load + accordion in Forge, byte-identical output with enable OFF in a real generation, ControlNet injection pixel effect (the 2026-10-07 A/B found the API route pixel-inert; the in-process path must re-prove), SDXL+Flux strength=1.0 similarity ≥ 0.55, and the alwayson_scripts API path are all unverified. The build labels these unverified correctly; per TASK-001 "Required checks" the task cannot be called done until the owner-4090 tunnel run happens. Included as a gate blocker so nobody presents offline green as done; owner may waive to an offline-only scope explicitly.
- [Major] F3 — Shipped tests do not cover the claims they are cited for. tests/test_logic.py (14 tests, re-run green in this pass) covers decide_mode mapping, detect_family, find_adapter_model, blend math, feather mask, template_embedding — matching the TASK-001 offline criterion as written. It does NOT cover: _inject_controlnet_unit / _build_controlnet_unit (the riskiest wiring; I verified it manually with stubs here — first disabled slot wins, all-busy returns False → loud downgrade — but that check is not in the repo), the enable-OFF no-op (covered only by check_disabled_noop.py in the workflow work dir, re-run green here, not shipped), or downgrade→infotext writing. Fix: port the no-op harness and a stubbed injection test into extensions/forge-face-consistency/tests/.
- [Major] F4 — Dead verify setting. settings.py OPT_VERIFY_THRESHOLD + swap_engine.SAME_PERSON_THRESHOLD (0.55) exist and a "verify threshold" is registered in Settings, but nothing consumes either: similarity is recorded, never compared, never warned on. A swap that lands at 0.20 similarity is reported identically to one at 0.93. Fix: write a "FaceConsistency verify: below threshold" infotext/log line when similarity_after < threshold, or remove the setting and the README implication.
- [Major] F5 — FaceSwapEngine (and template) rebuilt per image. postprocess_image constructs FaceSwapEngine() inside the per-image call; FaceAnalysis.prepare and build_template re-run for every image in a batch. With CPUExecutionProvider (swap_engine app/swapper) this makes batches pathologically slow. Fix: cache engine + template on the script instance or p, keyed by ref source. (Backing: swap_engine.py FaceSwapEngine.__init__/app lazy-init per instance; script postprocess_image instantiation site.)
- [Minor] F7 — README overclaims infotext contents for the ControlNet path: "Every run writes … ArcFace similarity before/after" — postprocess_image returns early for controlnet mode, so no similarity is written there. Either scope the sentence or record n/a explicitly.
- [Minor] F8 — Silent skip inside blended-swap: if no face is detected in the swapped image, blending is skipped and the full-strength swap is presented at strength < 1.0 with no log/infotext note (script postprocess_image, `if faces:` with no else). Add a log + infotext note on that branch (the module's own never-silent rule).
- [Minor] F9 — detect_family: a checkpoint filename hint containing "sdxl" without is_sdxl and without "xl"/"sdxl" in the class name resolves to "other" → safe loud downgrade, but a real SDXL checkpoint whose engine flag is unreadable would silently lose the reference path (still logged as downgrade, so acceptable; noted for the live check to confirm is_sdxl is populated).
- [Minor] F10 — Repo bookkeeping stale: .agents/status.json still phase=build, critic=null; TASK-001 header still "awaiting owner OK". Update both with this verdict.

## What passed clean (verified, not taken on faith)

- Enable OFF is a true no-op in code: before_process returns before touching extra_generation_params, postprocess_image/postprocess early-return; independent harness re-run printed DISABLED_NOOP_OK with pp.image identity preserved (live byte-identity still F6).
- Strength semantics: decide_mode → max=swap (not downgraded, every family), SDXL+adapter=controlnet at weight=strength, otherwise blended-swap with downgraded=True and a reason that is both logged and written to extra_generation_params/infotext.
- Flux guard: detect_family checks Flux first (wins over a stale is_sdxl flag — unit-tested) and decide_mode never routes Flux to controlnet.
- Injection mechanics match this clone: title match "ControlNet", args_from/args_to slice rewrite, first disabled slot; ControlNet reads units in process(), which Forge calls after before_process — ordering is sound.
- Multi-ref template: template_embedding drops refs < 0.35 cosine to the group mean (unit-tested with an opposing-identity outlier), build_template drives the swapper with the averaged embedding.
- Restore runs after swap: swap() does swapper.get(paste_back=True) then GFPGAN-or-detail-graft, reporting restored_by into infotext.
- No secrets, no real personal names, Forge tracked core untouched (git diff HEAD empty).

## Suggestions (research-backed)

- Cache the engine/template per run (F5) — backing: measured engine design in swap_engine.py (lazy app/swapper per instance) plus per-image instantiation site in the script; labelled partly opinion on magnitude until timed live.
- On the live run, test the SDXL-below-max injection with a deliberately wrong-face reference and diff pixels vs strength 0 — backing: the 2026-10-07 API A/B in .agents/research (ref-swap byte-identical via API) shows "accepted by infotext" is not evidence of effect; only a pixel diff is.

## Waivers
- none

---

# Critic re-review — TASK-001 (repair round 2) — 2026-10-07

- Reviewer: critic (independent re-review pass; did not write the repair)
- Reviewed: extensions/forge-face-consistency/ (scripts/forge_face_consistency.py, face_consistency/{logic,settings,swap_engine,blend,__init__}.py, tests/{test_logic,test_script_shell}.py, README.md), .gitignore diff, .agents/handoffs/TASK-001-repair-2-handoff.md, .agents/test-reports/TASK-001-offline-test.md
- Independence: all load-bearing claims re-verified firsthand in this pass — full source read, py_compile re-run (10 files, exit 0), pytest re-run (32 passed, exit 0), F1 exercised with a real PIL-encoded image (not stubs), git ignore/status/add -n re-run, import-purity re-run (IMPORT_PURITY_OK), Forge-core diff re-checked.
- Scope per bounded loop: only previously failed rows re-scored (Test, Correctness, Risk), plus regression check on the rest.

## Verdict
needs changes — one gate blocker remains (F6, pending live proof, not a code failure).

## Re-scored rubric rows

- Test: PASS (was FAIL/F3) — tests/test_script_shell.py ships 18 tests covering the enable-OFF no-op, ControlNet slot injection (first-disabled wins, dict units, all-busy → False, no-ControlNet → False), F1, downgrade→params, F4 verify warning both directions, F8 blend-skip, representative picking, threshold helper, template cache keys. Full suite re-run in this pass: 32 passed (14 pre-existing + 18 new), exit 0. No regression in the pre-existing 14.
- Correctness: PASS offline, UNVERIFIED live (was FAIL) — F1, F4, F5, F7, F8 all fixed on evidence below. Every acceptance criterion that needs a GPU remains under F6.
- Risk: FAIL on F6 only (was FAIL on F2+F6) — git-invisibility risk is resolved; the remaining risk is exactly the disclosed one: the SDXL below-max ControlNet injection has never been shown to move pixels on the owner's build (2026-10-07 API A/B was pixel-inert), and the swap similarity ≥ 0.55 is unproven live. Offline green must not be presented as done.

## Findings — disposition

- F1 (Blocking) FIXED — before_process now builds the ControlNet unit from representative_rgb_for_unit(): folder reference is decoded (logic.pick_representative_ref, sorted first image file), folder overrides the single image (matches UI wording), single image used when no folder. Verified firsthand with a real PIL image: folder-only returns an (8, 8, 3) uint8 ndarray (the old bug produced a 0-d object array from np.asarray(None)); empty/missing/corrupt-folder with no single image returns None → loud downgrade to blended-swap with the reason in plan and extra_generation_params["FaceConsistency downgrade"]. Shipped tests test_folder_only_reference_builds_unit_with_real_image / test_folder_without_images_downgrades_loudly / test_representative_prefers_folder_over_single_image pass.
- F2 (Blocking) FIXED — .gitignore is now /extensions/* + !/extensions/forge-face-consistency/ + !/extensions/forge-face-consistency/** (the only tracked change; git diff HEAD --stat = .gitignore alone, Forge core untouched — re-verified, so the "core untouched" evidence is no longer vacuous). git status --porcelain lists ?? extensions/forge-face-consistency/; git add -n lists all 11 real files (10 .py + README); other extension paths remain ignored (git check-ignore confirms extensions/some-other-ext/foo.py is still ignored).
- F6 (Blocking, gate) STILL OPEN — pending, not failed. No live evidence exists in the repo: extension load + accordion in real Forge, byte-identical output with enable OFF in a real generation, ControlNet injection pixel effect on the owner's build, SDXL+Flux strength=1.0 similarity ≥ 0.55, and the alwayson_scripts API path are all still unverified, and are honestly labelled as such in the task file, test report, and status.json. Per TASK-001 Required checks the task cannot be called done until the owner-4090 tunnel run, unless the owner explicitly waives to an offline-only scope. No waiver is recorded.
- F3 (Major) FIXED — the disabled-no-op harness and stubbed injection tests are shipped in extensions/forge-face-consistency/tests/test_script_shell.py and re-run green here.
- F4 (Major) FIXED — logic.is_below_threshold is consumed in postprocess_image: below-threshold similarity writes "FaceConsistency verify: below threshold (x < 0.55)", logs a WARNING, and adds an infotext note via postprocess(); None similarity reports n/a instead of warning. Both directions unit-tested and re-run green.
- F5 (Major) FIXED — engine cached on p via _get_engine (cleared per generation in before_process); FaceSwapEngine caches the averaged template per ref_source (_template_key: ndarray identity+shape, path string). Cache-key separation unit-tested.
- F7 (Minor) FIXED — README scopes the similarity claim to the swap paths; postprocess_image records similarity as "n/a (controlnet)" on the reference path instead of staying silent.
- F8 (Minor) FIXED — blend-skip (no face in swapped image) logs, writes "FaceConsistency blend skipped" to extra_generation_params, and appends the note to the infotext; unit-tested.
- F9 (Minor) unchanged — detect_family logic untouched; acceptable as before (real SDXL checkpoints losing the reference path would still be logged as a downgrade). Confirm is_sdxl is populated during the F6 live run.
- F10 (Minor) partially addressed by this pass — status.json updated to phase=review with F6 as the sole blocker; verdict posted to BOARD.md.

## New findings (repair round 2)

- [Minor] N1 — The F2 gitignore negation re-includes bytecode: !/extensions/forge-face-consistency/** overrides .gitignore line 3 (__pycache__), so git add -n lists 13 .pyc files under __pycache__/ dirs (verified: 24 add lines = 11 real files + 13 .pyc). A blanket git add would commit bytecode. Fix before commit: add an ignore for the extension's __pycache__ dirs after the negation (or delete the stray top-level extensions/forge-face-consistency/__pycache__/) and stage only real files. The test report's "git add -n lists all 10 files" is imprecise on this point (11 real files; 24 lines total).
- [Minor] N2 — Representative picking tries only the first sorted image file: if that file is corrupt but later folder files are fine, the run falls back to the single image or downgrades to blended-swap rather than trying the next file. Loud, never silent — acceptable; note for the live run.

## What passed clean (re-verified this pass)

- py_compile on all 10 extension .py files — exit 0.
- pytest: 32 passed under the workflow venv (pytest 9.1.1, numpy 2.5.3) — exit 0; the base_url warning is pre-existing and unrelated.
- Import purity: logic/settings/blend/swap_engine import with no gradio/torch/modules/insightface/onnxruntime/cv2 loaded — IMPORT_PURITY_OK.
- No secrets or real personal names in the extension (grep-verified); Forge tracked core untouched (git diff HEAD = .gitignore only).

## Waivers
- none. F6 stands until the owner-4090 live run or an explicit owner offline-only waiver.

---

# Critic re-review — TASK-001 (repair round 3) — 2026-10-07

- Reviewer: critic (independent re-review pass; did not write the repair)
- Reviewed: .gitignore diff, extensions/forge-face-consistency/face_consistency/{logic,__init__}.py, scripts/forge_face_consistency.py, tests/test_script_shell.py, .agents/handoffs/TASK-001-repair-3-handoff.md, .agents/tasks/TASK-001-forge-face-consistency.md
- Independence: all load-bearing claims re-verified firsthand in this pass — py_compile re-run (exit 0), pytest re-run (35 passed, exit 0), disabled-no-op harness re-run (DISABLED_NOOP_OK), import purity re-run (IMPORT_PURITY_OK), git add -n / check-ignore / diff re-run, waiver grep across task/review/status/board.
- Scope per bounded loop: confirm the previous blocker (F6) disposition, verify repair round 3 minors N1/N2, check for regressions against TASK-001.

## Verdict
needs changes — one gate blocker remains (F6, pending live proof, not a code failure). N1 and N2 are fixed; no regressions found.

## Findings — disposition

- F6 (Blocking, gate) STILL OPEN — not fixed, not waived. Confirmed firsthand: no live evidence exists in the repo (no tunnel-run report, no similarity measurement, no live infotext), and a grep for waiver/offline-only across TASK-001, this review, status.json, and BOARD.md finds only statements that F6 is pending — no owner waiver is recorded. All five sub-criteria remain unverified: (1) extension loads in real Forge + accordion in txt2img/img2img, (2) byte-identical output with enable OFF in a real generation, (3) ControlNet injection pixel effect on the owner's build (the 2026-10-07 API A/B was pixel-inert; the in-process path must re-prove), (4) SDXL + Flux strength=1.0 ArcFace similarity ≥ 0.55, (5) alwayson_scripts API path. TASK-001 acceptance checkboxes remain all unchecked and Required checks mark the live run "(unverified until he runs Forge)" — honestly labelled. Per TASK-001 the task cannot be called done until the owner-4090 tunnel run, unless the owner explicitly waives to an offline-only scope.
- N1 (Minor, round 2) FIXED — .gitignore now adds `/extensions/forge-face-consistency/**/__pycache__/` and `/**/*.pyc` after the negations. Verified firsthand after pytest regenerated caches on disk: `git add -n extensions/forge-face-consistency` lists exactly 11 files (10 .py + README), 0 `.pyc`; `git check-ignore -v` shows the pycache rule matching and `extensions/some-other-ext/foo.py` still ignored; `git diff HEAD --stat` = `.gitignore` only (Forge core untouched).
- N2 (Minor, round 2) FIXED — `logic.list_ref_candidates` returns all sorted image paths; `representative_rgb_for_unit` tries each candidate until one decodes, names skipped unreadable files in the source note, and an all-corrupt folder still falls back to the single image or returns None → the existing loud blended-swap downgrade. 3 new tests (candidates sorted/filtered, corrupt-first-file skipped, all-corrupt returns None) pass.

## Regression check (against TASK-001)

- py_compile on all extension .py files — exit 0.
- pytest: 35 passed (14 test_logic + 21 test_script_shell; 32 pre-existing + 3 new), exit 0; the base_url warning is pre-existing and unrelated. No pre-existing test regressed.
- Disabled-no-op harness re-run: DISABLED_NOOP_OK. Import purity: IMPORT_PURITY_OK (no gradio/torch/modules/insightface/onnxruntime/cv2).
- No secrets or real personal names in the extension (grep-verified); diff scoped to .gitignore + the untracked extension (Forge tracked core untouched).
- Previously fixed F1–F5/F7/F8 untouched by round 3 and covered by the re-run green suite.
- Bookkeeping note (minor, non-blocking): .agents/status.json still says phase=build; it does carry F6 as the sole blocker. Update phase to review at the next builder touch.

## Waivers
- none. F6 stands until the owner-4090 live run or an explicit owner offline-only waiver.
