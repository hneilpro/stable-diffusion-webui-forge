# Test report — TASK-002 — 2026-10-08

- Branch: feat/forge-consistent-character
- Changed behavior: API (txt2img/img2img) requests now default to saving generated images to the output folders; extension README documents the two GFPGAN files with exact URLs and folders. (TASK-001 repair-round-6 changes in the same worktree were pre-existing and untouched.)

## Commands run
- `python3 -m py_compile modules/api/models.py modules/api/api.py extensions/forge-face-consistency/scripts/forge_face_consistency.py` — OK — exit 0
- AST check (parse `modules/api/models.py`, assert both `save_images` additional-field defaults are `True`; assert `api.py` still maps `do_not_save_samples`/`do_not_save_grid` from the flag for both endpoints) — `save_images defaults in models.py: [True, True]`, `api.py opt-out mapping intact: OK` — exit 0 (first run of this check had a bug in the check itself — it looked for `save_images` among dict keys instead of the `key` entry's value; fixed the check, code was already correct)
- `work/test-env/venv/bin/python -m pytest extensions/forge-face-consistency/tests/ -q` — 45 passed, 1 pre-existing config warning (`Unknown config option: base_url`) — exit 0

## Live / manual checks
- None possible in this VM (no GPU, no tunnel to the owner's Forge). Live criteria stay unverified (below).

## Outputs inspected
- `git diff modules/api/models.py` — only the two `save_images` defaults flipped `False` -> `True`, each with a two-line comment; no other API field touched.
- `git diff extensions/forge-face-consistency/README.md` — Requirements bullet expanded into a "GFPGAN — two different files" section: extension ONNX (`gfpgan_1.4.onnx`, FaceFusion models-3.0.0 HF URL -> `models/insightface/gfpgan_1.4.onnx`, infotext confirmation `restored by: gfpgan`) vs Forge-native (`GFPGANv1.4.pth`, TencentARC v1.3.0 release URL -> `models/GFPGAN/`). URLs and folders cross-checked against `modules/gfpgan_model.py` (model_url), `modules/cmd_args.py:55` (default `models/GFPGAN`), and `face_consistency/swap_engine.py` `default_model_candidates`.
- Save path traced end-to-end in source: `api.py` sets `do_not_save_samples/do_not_save_grid = not save_images`; `processing.py:513-515` saves only when `opts.samples_save` (default True) and `not do_not_save_samples`; outputs land in `opts.outdir_txt2img_samples`/`outdir_img2img_samples` (`api.py` assigns `p.outpath_samples` from those opts).

## Failures
- None caused by this change. The 45-test suite includes TASK-001's repair-round-6 test (`test_injection_rewrites_tuple_script_args`); all green.

## Skipped / unverified
- Live API save on the owner's PC — unverified — needs his Forge running this branch + one API txt2img call without `save_images`; expect a PNG under `outputs/txt2img-images/`. An explicit `"save_images": false` call should write nothing.
- Live GFPGAN ONNX restore in Forge — unverified — after he places `gfpgan_1.4.onnx` and restarts, a max-strength swap's infotext should read `FaceConsistency restored by: gfpgan`.
- Forge-native GFPGAN panel/restore — unverified — same PC session (Settings face-restoration model = GFPGAN, one gen or Extras run).

## Coverage gaps
- No repo-level API test suite exists upstream, so the new default is covered by the AST/source check + the traced save path, not an HTTP-level test. Acceptable for a two-line default flip; the live check above closes the gap on his PC.
