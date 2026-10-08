# Handoff — TASK-001 repair round 4 (API string reference) — 2026-10-07

- Role: builder -> critic (separate pass) / owner (live retest)
- Changed files:
  - extensions/forge-face-consistency/scripts/forge_face_consistency.py
    (_decode_ref_string; _ref_to_rgb/_ref_to_bgr accept str/bytes/dict/
    path/base64, return None instead of raising on undecodable input)
  - extensions/forge-face-consistency/tests/test_script_shell.py (+7 tests)
  - .agents/test-reports/TASK-001-live-test.md (live round 1 evidence)
- What happened: first live run on the owner's 4090 proved the extension
  loads and OFF is byte-identical, but every enabled API run failed at
  first contact with the reference image: the API sends image script
  args as base64 strings and _ref_to_rgb indexed shape[-1] on the 0-d
  array np.asarray(str) makes. Owner's PC traceback confirms the exact
  line (117). Fix + tests above; offline suite 42 passed, py_compile OK.
  The extension's swap engine was exercised agent-side on a real live
  output: ArcFace 0.049 -> 0.921, so the swap core is not in question.
- Known risks / unverified: the fix has only run in this VM. The owner's
  PC must pull the branch and restart Forge before the F6 retest (SDXL
  swap >= 0.55, below-max injection pixel diff, Flux swap) can run.
- Next: owner approved commit + push (2026-10-07, this commit) -> owner
  pulls + restarts Forge -> builder re-runs the live matrix -> critic
  re-scores F6 only.
