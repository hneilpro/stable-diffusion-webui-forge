# Handoff — TASK-001 repair round 5 (setup self-reporting) — 2026-10-08

- Role: builder -> critic (separate pass) / owner (live retest round 3)
- Changed files:
  - extensions/forge-face-consistency/scripts/forge_face_consistency.py
    (whole before_process setup block wrapped: on any exception, print
    traceback, record "FaceConsistency error" in the infotext, downgrade
    loudly to blended swap; mode/strength/family params now always land)
  - extensions/forge-face-consistency/tests/test_script_shell.py (+2 tests)
  - .agents/test-reports/TASK-001-live-test.md (live round 2 evidence)
- Live round 2 verdict (owner 4090, commit 2dc41d3): SDXL swap at max
  PASSES in Forge — extension-measured and independently re-measured
  ArcFace 0.938 >= 0.55; detail-graft restore leaves a visible seam at
  jaw/hairline (no GFPGAN ONNX installed); the alwayson API path passes
  with it. Below-max injection still failed silently: the unguarded
  setup block raised before writing mode/family params, so the plan
  stayed "disabled" and E/F came out pixel-identical to baseline; the
  exact exception lives only in the Forge console.
- Flux is blocked on the PC itself, not by the extension: txt2img with
  the Flux checkpoint returns 500 "You do not have CLIP state dict!"
  (backend/loader.py). The owner's screenshots show clip_l + t5xxl_fp8
  already in models/text_encoder/ and ae.safetensors in models/VAE/ —
  Forge simply had no additional modules attached (additional_modules:
  []). Retest will set the forge_additional_modules option via the API
  (clip_l, t5xxl_fp8, ae) before generating, then restore [].
- Next: pushed on owner go (2026-10-08, this commit) -> owner pulls +
  restarts Forge -> builder retest round 3 (injection must now
  self-report its exception and/or engage; Flux swap with encoders
  attached) -> critic re-scores F6.
