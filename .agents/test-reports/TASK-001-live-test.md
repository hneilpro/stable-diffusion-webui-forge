# Test report — TASK-001 live round 1 — 2026-10-07

- Branch: feat/forge-consistent-character
- Scope: first live run of extensions/forge-face-consistency on the owner's
  RTX 4090 Forge via a pasted tunnel URL (URL not recorded, per repo rules).
- Tunnel generation outputs: ~/workspace/forge-live-test/out/ (agent VM).

## Server probe

- Forge reachable; checkpoints: epicrealismXL_vxviiCrystalclear (SDXL,
  loaded) and iniverseMixSFWNSFW_f1dRealnsfwGuofengV2 (Flux).
- GET /sdapi/v1/script-info lists "forge face consistency" as alwayson for
  both txt2img and img2img, args [enabled=false, ref=null, refs_dir=null,
  strength=0.85, restore=true] — the extension loads in real Forge.
- ControlNet models include ip-adapter-faceid-plusv2_sdxl and the InstantID
  pair — the below-max SDXL reference path has an adapter available.

## Live generations (512x768, 20 steps, DPM++ 2M Karras, CFG 6, seed 424242)

Prompt: head-and-shoulders studio portrait of a woman. Reference: the
canonical character face (agent-side file, not in this repo).

| run | args | result |
| --- | --- | --- |
| a_off_noscript | no alwayson script | sha256 cd19d50d1250adbf… |
| b_off_disabled | script present, enabled=false | sha256 cd19d50d1250adbf… — byte-identical to A |
| c_sdxl_swap_max | enabled, strength 1.0 | pixel-identical to A; infotext "FaceConsistency error: tuple index out of range", mode=swap |
| g_sdxl_strength0 | enabled, strength 0.0 | pixel-identical to A |
| e_sdxl_ref_vj | enabled, strength 0.85, character ref | pixel-identical to A and to F; infotext mode=disabled |
| f_sdxl_ref_wrong | enabled, strength 0.85, different-person ref | pixel-identical to A and to E |

Pixel comparison (PIL, decoded pixels, not file bytes): every run's
mean |diff| vs A = 0.000. Independent ArcFace verify (agent-side
InsightFace) of C vs the reference: 0.049 — identical to A's 0.049. The
swap never ran; no ControlNet injection ever happened.

## Root cause (confirmed by the owner's PC console)

Owner-pasted traceback: before_process -> representative_rgb_for_unit ->
_ref_to_rgb, line 117 `if arr.shape[-1] == 4:` IndexError: tuple index
out of range. The API delivers image script args as base64 strings; the
script assumed the WebUI's numpy array. np.asarray(str) is a 0-d array,
so shape[-1] raises. Reproduced exactly in isolation on this VM.
before_process crashed before writing mode/params (hence mode=disabled
for E/F), and for the swap run _ref_to_bgr died the same way inside
postprocess_image, where the existing try/except recorded it in the
infotext. The API path was therefore exercised but ref handling was not
API-safe — an F6 sub-criterion found live, exactly as the gate intended.

## What passed live

- Extension loads in Forge; accordion script registered for txt2img and
  img2img (script-info). UI accordion rendering itself not visually
  inspected (API-only session).
- Enable OFF is byte-identical to no-script on a real SDXL generation.

## Fix (repair round 4, this VM; not yet on the owner's PC)

- scripts/forge_face_consistency.py: _ref_to_rgb/_ref_to_bgr now accept
  numpy, PIL, dict, bytes, and strings (base64 / data-URI / file path)
  via _decode_ref_string; undecodable input returns None instead of
  raising; ndim/size validated before shape indexing.
  representative_rgb_for_unit and the swap ref path inherit this.
- tests/test_script_shell.py: +7 tests (base64/data-URI/path decode,
  garbage -> None not crash, BGR channel flip, representative from a
  string ref, before_process with a string ref injecting a ControlNet
  unit with a real ndarray, string ref at max setting the swap ref).
- Offline evidence: pytest 42 passed (35 pre-existing + 7 new), exit 0;
  py_compile 10/10 exit 0 (venv: pytest 9.1.1, numpy 2.5.3, pillow 12.3.0).
- Engine de-risk: the extension's own FaceSwapEngine, run agent-side on
  live output A, moved ArcFace similarity 0.049 -> 0.921 (GFPGAN restore),
  target out/local_engine_swap_check.png in the agent live-test dir. The
  swap engine works; only the ref decode blocked it in Forge.

## Still unverified (F6 remainder)

- SDXL strength=1.0 swap similarity >= 0.55 in Forge (after the owner's
  PC pulls the fix and Forge restarts).
- SDXL below-max ControlNet injection pixel effect (VJ ref vs wrong ref
  vs strength 0, same seed) in Forge.
- Flux strength=1.0 swap similarity >= 0.55.
- Arg order via alwayson_scripts is confirmed accepted (script-info +
  runs executed), but the enabled-path API run needs the retest above.

---

# Live round 2 — 2026-10-08 (after repair round 4, commit 2dc41d3)

Same matrix, fresh tunnel (URL not recorded). Outputs:
~/workspace/forge-live-test/out2/.

| run | result |
| --- | --- |
| a2_off | SDXL baseline, sha 75772ed28e82 |
| c2_sdxl_swap_max | mode=swap in infotext; extension-measured similarity 0.039 -> 0.938; restored by detail-graft; independent agent-side verify 0.938; 12.6% of pixels differ from baseline (>16/255). Viewed: face matches the reference; visible paste seam at the jaw/hairline from the detail-graft restore (no GFPGAN ONNX on the PC). PASS (>= 0.55) |
| g2_strength0 | pixel-identical to baseline, mode=disabled recorded |
| e2_ref_vj (0.85) | pixel-identical to baseline; infotext mode=disabled, no mode/family params — the setup block still raised before writing params |
| f2_ref_wrong (0.85) | identical to e2_run — no injection |
| Flux gens | checkpoint switch OK; txt2img returns HTTP 500 {"error":"AssertionError","message":"You do not have CLIP state dict!"} even with the extension absent — his Flux checkpoint carries no CLIP/T5 encoders and Forge has none loaded. Flux generation itself is blocked on the PC until encoder files are installed; not an extension failure. SDXL checkpoint re-selected afterwards and verified via /sdapi/v1/options. |

## Repair round 5 (this VM; not yet committed)

Round 2's injection failure sat in the unguarded setup block: build/
inject of the ControlNet unit raised, Forge logged to its console,
generation continued with the plan at mode=disabled and nothing in the
infotext. Fix: the whole setup block is wrapped — on any exception the
script prints the traceback, records "FaceConsistency error" in the
infotext, and downgrades loudly to blended swap so the next live run
(self-)reports the exact exception over the API. Params (mode/strength/
family) are now written even on that path. +2 offline tests (build
failure, family-probe failure -> recorded blended-swap downgrade);
suite 44 passed, py_compile OK.

---

# Live round 3 — 2026-10-08 (after repair round 5, commit 6b72358)

Outputs: ~/workspace/forge-live-test/out3/. The self-reporting works:
the infotext now carries the exact exception.

| run | result |
| --- | --- |
| a3_off | baseline; sha identical to round 2's baseline across restarts |
| e3_ref_vj (0.85) | mode=blended-swap via the new loud downgrade; infotext "FaceConsistency error: 'tuple' object does not support item assignment"; extension-measured 0.039 -> 0.938, independent verify 0.902; 8.0% of pixels differ from baseline. Root cause found: Forge hands p.script_args as an immutable tuple and _inject_controlnet_unit assigned into it |
| f3_ref_wrong (0.85) | same downgrade; 24.4% of pixels differ from baseline, 27.1% from e3 — the fallback is reference-dependent, as designed |
| g3_strength0 | pixel-identical to baseline |
| Flux (encoders attached via forge_additional_modules: clip_l, t5xxl_fp8, ae — accepted, stored as full paths) | flux_off hit a Cloudflare 524 on the tunnel during the first slow load; flux_swap then returned in 68 s but produced a pure black image (1,890-byte PNG; no face detectable, extension recorded "no face detected in target image"). Forge options were restored to SDXL + no modules afterwards |

Repair round 6 (this VM): _inject_controlnet_unit converts
p.script_args to a list before assigning (+1 tuple test; 45 tests
pass). The ControlNet injection itself is still unproven until the
retest. Flux black-frame is a generation-side issue (sampler/guidance
fit for this distilled Flux fine-tune) to retry with different settings
on the next run; the extension correctly recorded the failure instead
of swapping nothing silently.

## F6 status after round 2

- Loads (txt2img/img2img): PASS (round 1).
- OFF byte-identical: PASS (round 1).
- API alwayson path: PASS for the swap route (c2 ran entirely through
  alwayson_scripts).
- SDXL swap >= 0.55: PASS (0.938, measured twice).
- SDXL below-max injection pixel effect: FAIL so far — root-cause text
  pending the round-5 self-report run (or the owner's console paste).
- Flux swap >= 0.55: BLOCKED on the PC (no CLIP state dict); needs
  text-encoder files installed or an owner waiver for this sub-point.
