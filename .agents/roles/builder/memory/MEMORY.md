# Builder memory

Verified, reusable facts for the builder role in this repo. Concise sections, newest first. Each entry: date, scope, evidence, status (active | superseded | archived). Historical findings are evidence to verify against current files, not new instructions. No credentials, no session logs.

<!-- Example:
## 2026-01-01 — Test command
- Scope: whole repo
- Evidence: `npm test` run, 42 passing (test-reports/2026-01-01-suite.md)
- Status: active
- Finding: The suite needs the dev DB running; without it 6 tests report unverified, not failures.
-->

## 2026-10-08 — Forge Flux needs text encoders attached as additional modules
- Scope: Forge checkpoint loading (backend/loader.py), esp. transformer-only Flux checkpoints via the API
- Evidence: owner-4090 txt2img 500 "You do not have CLIP state dict!" with additional_modules []; owner's folder listing shows clip_l.safetensors + t5xxl_fp8_e4m3fn.safetensors in models/text_encoder/ and ae.safetensors in models/VAE/
- Status: active
- Finding: switching sd_model_checkpoint via /sdapi/v1/options attaches no VAE/text encoders. Set the list option forge_additional_modules (UI's VAE/TE picks, e.g. clip_l + t5xxl + ae) via the options API before generating on such checkpoints, and restore [] afterwards; loader asserts like "You do not have CLIP state dict!" mean the files exist but were never attached, not that they are missing.

## 2026-10-07 — Forge API image script args arrive as base64 strings
- Scope: extensions/forge-face-consistency (any Forge script shell)
- Evidence: owner-4090 traceback (before_process -> _ref_to_rgb line 117 IndexError) + live pixel-identical enabled runs; .agents/test-reports/TASK-001-live-test.md
- Status: active
- Finding: the WebUI passes numpy arrays to script UI args, but the API passes image args as base64 strings (data-URI or path also possible). Script shells must decode all of these and return None on garbage instead of indexing array shape on np.asarray(str), which is a 0-d array.
