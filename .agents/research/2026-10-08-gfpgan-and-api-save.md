# Research — GFPGAN models + API save-by-default — 2026-10-08

- Task: TASK-002
- Question: (1) Which file(s) does the owner need on his PC so GFPGAN actually restores faces, and where do they go? (2) What is the smallest change that makes API generations save to the output folder by default?

## Findings
- Two different "GFPGAN" models exist in this repo and do not substitute for each other:
  - Forge native (PyTorch, spandrel): `GFPGANv1.4.pth`. Loaded by `modules/gfpgan_model.py` (`FaceRestorerGFPGAN`, model_url = TencentARC GFPGAN v1.3.0 release asset `GFPGANv1.4.pth`) from the `--gfpgan-models-path` directory (default `models/GFPGAN`, `modules/cmd_args.py:55`; wired in `modules/initialize.py:65`). Used by built-in "Restore faces" (`opts.face_restoration_model`) and the Extras GFPGAN post-processor (`scripts/postprocessing_gfpgan.py`). If the file is missing, `load_net` falls back to downloading it from the official release URL.
  - Face-consistency extension (ONNX): `gfpgan_1.4.onnx` (FaceFusion models-3.0.0 asset). `face_consistency/swap_engine.py` `default_model_candidates` searches `models/insightface/gfpgan_1.4.onnx` (next to `inswapper_128.onnx`), then `extensions/forge-face-consistency/models/gfpgan_1.4.onnx`. `gfpgan_restore` runs it via onnxruntime on the aligned 512 crop. This is the file whose absence on the owner's PC made live round 2 fall back to `restored_by: detail-graft` (visible seam).
- The extension's ONNX file is already proven agent-side: local `~/workspace/forge-face-swap/models/gfpgan_1.4.onnx` (325M, ONNX I/O `[1,3,512,512]` verified 2026-10-07); swap engine scored 0.049 -> 0.921 with it. Owner-PC step is just: download `gfpgan_1.4.onnx` (~340MB) from `https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx`, place at `<Forge>/models/insightface/gfpgan_1.4.onnx`, restart Forge. Confirmation signal: infotext `FaceConsistency restored by: gfpgan`.
- API save behaviour: `modules/api/models.py:106,125` define `save_images` with default `False` for both txt2img and img2img API models. `modules/api/api.py:458-459,527-528` map it to `do_not_save_samples/do_not_save_grid = not save_images`, and `modules/processing.py:513-515` only saves when `opts.samples_save` (default True, `modules/shared_options.py:37`) and `not do_not_save_samples`. So flipping the two API defaults to `True` makes API runs save samples (and grids, same flag) into `opts.outdir_txt2img_samples`/`outdir_img2img_samples` exactly like UI runs, while still honouring an explicit `"save_images": false` opt-out per request. No core save-path change is needed.

## Sources
- Repo code read 2026-10-08: `modules/gfpgan_model.py`, `modules/cmd_args.py`, `modules/initialize.py`, `scripts/postprocessing_gfpgan.py`, `extensions/forge-face-consistency/face_consistency/swap_engine.py`, `modules/api/models.py`, `modules/api/api.py`, `modules/processing.py`, `modules/shared_options.py`.
- Web search 2026-10-08 for the ONNX distribution point: FaceFusion `models-3.0.0` release asset `gfpgan_1.4.onnx` on Hugging Face (multiple independent repos cite the same resolve URL).
- `.agents/test-reports/TASK-001-live-test.md` (round 2: "no GFPGAN on the PC", detail-graft restore).

## Conflicts / uncertainty
- None on paths or URLs. The only unverified piece is behaviour on the owner's PC after his download/restart — needs a live run with a fresh tunnel URL (same F6-style gate as TASK-001). Offline checks cannot prove the ONNX session or the API write path on his install.

## Decision unlocked
- GFPGAN needs no code change: instructions only (two files, two folders; the extension needs the `.onnx`, Forge-native restore needs the `.pth`). Document the exact steps in the extension README.
- API save-by-default is a two-line default flip in `modules/api/models.py` (TASK-002 build). Alternatives rejected in the task plan.
