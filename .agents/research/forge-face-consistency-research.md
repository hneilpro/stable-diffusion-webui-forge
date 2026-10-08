# Research — Forge face consistency (SDXL + Flux)

Date: 2026-10-07. Repo: hneilpro/stable-diffusion-webui-forge, branch feat/forge-consistent-character.

## What Forge already ships (verified in this clone)
- `extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py` registers ControlNet preprocessors:
  CLIP-ViT-H / CLIP-ViT-bigG (IP-Adapter), InsightFace+CLIP-H (IP-Adapter FaceID), InsightFace (InstantID),
  plus `IPAdapterPatcher` (ControlModelPatcher) that applies IP-Adapter inside sampling
  (`process_before_every_sampling`, weight = unit strength). FaceID v2 flag from filename.
- `extensions-builtin/forge_preprocessor_reference` — reference-attention preprocessor (style fidelity slider).
- `extensions-builtin/sd_forge_controlnet` — ControlNet units carry weight/guidance into the patchers.
- Script API (`modules/scripts.py`): `Script.ui()`, `before_process()`, `postprocess_image(p, pp)` with
  `PostprocessImageArgs(image, index)` — the post-process hook ReActor-style swaps use.

## Routes to identity, by family
- SDXL reference (during diffusion): IP-Adapter FaceID Plus v2, InstantID, PuLID-SDXL. All InsightFace-based.
  Forge supports the first two natively via ControlNet (above). FaceID needs its paired LoRA to work well.
- Flux reference: PuLID-Flux is the real route; IP-Adapter FaceID has no Flux variant. Forge ships no PuLID.
- Face swap (after diffusion): InsightFace inswapper_128.onnx (the ReActor model). Pure pixel post-process,
  so it is model-agnostic — works identically for SDXL, Flux, SD1.5. This is how Forge+Flux face swap is
  done in practice (ReActor extension). Cost: swapper renders at 128px, faces come back soft without a
  restoration pass.

## Our proven engine (from the earlier wrong-repo work, to port)
`~/workspace/forge-face-swap/swap.py` (220 lines, live-measured on his 4090 outputs):
- buffalo_l detection + inswapper swap, multi-reference averaged template embedding with outlier
  cleaning (refs <0.35 cosine to the mean are dropped) — steadier than any single photo.
- ArcFace cosine verify before/after (drifted img2img ~0.10 -> swapped 0.92-0.94; >=0.55 = same person).
- Restoration: GFPGAN 1.4 ONNX when present, else high-frequency detail graft inside a feathered face
  mask (128px swap measured sharpness 99.5 -> 22.8 without it).
- Ceiling, stated plainly: a swap pastes identity onto an existing head — it cannot fix wrong
  geometry/expression. Conditioning during generation forms the geometry as the person and is better
  below max strength when the backend truly works.

## Useful findings to integrate
1. Multi-ref averaged template (above) as the identity source, not a single photo.
2. Verify gate: report before/after ArcFace scores in infotext; never present an accepted payload as a likeness.
3. Restore after swap (GFPGAN/detail graft) — mandatory, not optional.
4. Family guard: never send SDXL adapters to a Flux checkpoint (Flux -> swap path by default).
5. Use the ReActor model-path convention `models/insightface/inswapper_128.onnx` so existing installs work.
6. RefDrop (sd-refdrop-forge) exists for Forge consistency via recorded reference features — related work,
   different mechanism (needs a liked seed/output first); not integrated, noted for later.

## Design consequence
One control the user asked for: Enable + Reference Strength slider.
- Strength < max: face reference. SDXL with a FaceID/InstantID ControlNet model present -> inject a
  ControlNet unit at weight=strength. Otherwise (or Flux) -> blended swap, blend=strength, feathered mask.
- Strength = max (1.0): full face swap post-process, then restore. Model-agnostic guarantee.
Every path logs what it actually did (mode used + scores) in infotext — downgrades are reported, never silent.
