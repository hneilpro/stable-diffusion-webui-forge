# Forge Face Consistency

One control — **Enable + reference image + Reference Strength** — keeps a
chosen face consistent across SDXL and Flux generations in
[Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

| Strength | Behaviour |
| --- | --- |
| `1.0` (max) | Full InsightFace face swap (`inswapper_128`, `paste_back`) + face restoration on the generated image. Works for SDXL, Flux, and any other checkpoint. |
| `< 1.0`, SDXL, FaceID/InstantID ControlNet model installed | Injects a ControlNet unit (IP-Adapter FaceID or InstantID) at `weight = strength` — identity is formed during diffusion. |
| `< 1.0`, otherwise | Blended swap: full swap composited inside a feathered face mask at `blend = strength`. This is the honest fallback for Flux (Forge ships no Flux FaceID adapter) and for SDXL without an adapter model. |

Every run writes the mode actually used (`faceid` / `instantid` /
`blended-swap` / `swap` / `outfit-only`) and the strength into the image infotext. The
swap and blended-swap paths also record the ArcFace similarity
before/after, the target face's share of the image, a head-height
proportion assessment (`FaceConsistency head height`, with a warning
when the head is implausibly small for the frame), plus a warning when
the result lands below the verify threshold (Settings → Forge Face
Consistency). The ControlNet reference paths form identity during
diffusion, so their similarity is recorded as `n/a (controlnet)` rather
than measured. A downgrade (e.g. Flux below max, or no adapter found) is
always recorded — never silent.

Face detection runs at 640px and automatically re-tries at 1280px for
large frames where no face was found (full-body generations). When
several reference photos are given, the sharpest detectable face drives
the swapper and the ControlNet unit — not the first file alphabetically.
InsightFace and GFPGAN run on CUDA when `onnxruntime-gpu` is present,
falling back to CPU otherwise.

## Character LoRA — preserving body type, not just the face

Face swap and FaceID/InstantID only carry the *face*; the body comes
from the prompt and the base model. To keep the same body type across
poses and outfits, train a small character LoRA on 6–20 photos of the
person (varied backgrounds, lighting, and clothing; include half-body
shots), then set **Settings → Forge Face Consistency → Character LoRA
name** (a file in `models/Lora`, without extension) and **weight**
(0.6–0.8 is a good start). The script appends
`<lora:name:weight>` to the prompt on every enabled run, so identity
(face + body) is baked into the diffusion itself and the swap/FaceID
path locks the face on top.

Practical training recipe (SDXL, single 24 GB GPU): rank 16–32,
learning rate 1e-4, ~100–200 steps per image, bf16 — about 10–30
minutes with kohya sd-scripts or ai-toolkit. Name the file after the
character (e.g. `tori_xl.safetensors`) and put that name in the setting.

## Outfit / object reference — dress the character from a photo

A second, independent control — **Reference outfit / object** plus
**Outfit reference strength** — injects a general IP-Adapter ControlNet
unit (Forge's built-in `sd_forge_ipadapter`), so a photo of an outfit, a
prop, or an environment steers the generation. It works at any face
strength (including a max-strength face swap) because it occupies its
own ControlNet slot, and it also works with no face reference at all
(`outfit-only` mode). A well-composed reference photo transfers
composition and body proportions too, which is the most reliable fix
found for small-head full-body renders.

Setup:

1. Download `ip-adapter_sdxl.safetensors` from
   `h94/IP-Adapter` (`sdxl_models` folder) and save it in
   `models/ControlNet/` (filename must contain `ip-adapter`).
2. Restart Forge. The CLIP vision encoder (`CLIP-ViT-bigG`) downloads
   automatically on first use.
3. In the accordion, drop an outfit/object photo into **Reference outfit
   / object** and set **Outfit reference strength** to 0.4–0.7
   (0.5 balances the photo against your prompt; higher follows the
   photo more literally). 0 = off.

Notes:

- SDXL and SD1.5 only. Flux has no IP-Adapter in this Forge build, so
  the outfit reference is skipped with a logged reason on Flux.
- If no IP-Adapter model is found, the run continues without it and
  the infotext says exactly which file to download — never silent.
- The same mechanism accepts any reference subject: a jacket, a
  robotic arm, a desk setup — anything you want carried into the scene.

## Body proportions — what actually works

Measured finding (Oct 2026, SDXL photoreal checkpoints): extreme
full-body framing (e.g. 768×1152, head-to-feet) renders heads too small
to look right, and no prompt wording fully corrects it. What works:

- **3/4 framing** (head to knees, 768×1024): the head lands at a
  natural size. This is the recommended framing for character work.
- **Outfit/style reference** (above) with a well-proportioned photo:
  the IP-Adapter conditioning transfers composition as well as
  garments.
- Prompt grounding helps at the margin: `realistic natural
  head-to-body proportions`, and negative `small head, elongated body`.
- The extension measures every swapped face and writes
  `FaceConsistency head height` (head height as % of frame) into the
  infotext, warning when it drops below 7% — an implausibly small head
  for any framing with a visible face.

### Body-proportion gate (automated)

The face-similarity check only verifies the *face*; this gate checks
the *body*. On every enabled generation it runs OpenPose (via the
ControlNet annotator, in-process) on the finished image, reduces the
pose to five proportions — body height in heads, shoulder:hip ratio,
shoulder and hip width in heads, leg fraction of body height — and
compares them against the reference profile under **Settings → Forge
Face Consistency** (defaults are one specific person's measured
proportions; change them for anyone else). The measured values are
always written to the infotext (`FaceConsistency body`).

Three modes:

- `off` — the gate does nothing.
- `warn` (default) — deviations beyond the tolerance (default ±15%)
  are written as `FaceConsistency body gate: WARNING: ...` in the
  infotext and the console log.
- `reject` — a failing body **aborts the generation** with a
  `BodyProportionError` naming the offending ratios, before the image
  is presented.

The gate runs in every face mode (swap, blended-swap, FaceID/InstantID
ControlNet, outfit-only) because the body comes from diffusion in all
of them — only the face is ever swapped. If the pose detector is
unavailable or no pose is found, the gate degrades to a loud infotext
note, never a silent skip. Waist:hip is deliberately not measured:
keypoints carry no waist landmark, and guessing it would be dishonest.

### Torso consistency — chest and belly button

No AI generator pins chest or navel identity specifically; every
system works at whole-character granularity. What this extension does
instead, on top of the body-proportion gate:

- **Torso reference slot** (Settings → Forge Face Consistency →
  `ffc_torso_ref_dir`, plus the per-generation *Torso reference
  strength* slider, default 0.45). A general IP-Adapter ControlNet
  unit conditioned on a square torso crop of the first readable image
  in the folder — pose-based crop when the annotator finds
  shoulders+hips, center square crop otherwise. Must be the
  *general/Plus* adapter; FaceID variants are face-only by
  architecture and are excluded by the picker, exactly like the
  outfit slot. Same model the outfit slot needs
  (`ip-adapter_sdxl.safetensors` in `models/ControlNet`); Flux skips
  loudly.
- **Depth lock** (optional, `ffc_torso_depth_weight`, default 0 = off).
  A depth ControlNet unit on the same torso crop, for geometric
  anchoring of breast volume and waist curve. Needs a depth ControlNet
  model in `models/ControlNet`; missing model or busy slots degrade
  to a loud infotext note.
- **Navel gate** (warn-only, `ffc_navel_gate`, default `warn`). The
  expected navel pixel is computed from the pose keypoints via the
  torso standard (midline, 15% of the way from waist to crotch); the
  detected navel comes from template-matching a reference navel crop
  (`ffc_navel_template`) inside the waist ROI with normalized
  cross-correlation. Deviations beyond 0.20 head heights warn in the
  infotext (`FaceConsistency navel`). There is deliberately no reject
  mode: the honest ceiling is stable navel *position*, never
  pixel-identical identity.
- **Navel detailer** (optional, `ffc_navel_detailer`, default off).
  After generation, the navel ROI is cropped, upscaled 3×, re-rendered
  with a low-denoise (0.35) img2img pass, and feathered back in — the
  same crop/upscale/img2img/paste-back pattern as the hand fix. This
  stabilizes navel position and cleans up rendering; it regenerates
  the navel rather than transplanting identity.

## Use

1. Open the **Forge Face Consistency** accordion (txt2img or img2img).
2. Tick Enable, drop in a reference face photo (or point *Reference images
   folder* at a folder of photos of the person — they are averaged into one
   outlier-cleaned identity template). For the ControlNet reference path,
   the folder's image files are tried in sorted order until one decodes.
3. Set strength: max for a guaranteed swap, lower for a guided reference.
4. Optionally, drop an outfit/object photo into *Reference outfit /
   object* and set its strength (see above).

Global defaults (enable, strength, restore, verify threshold, inswapper
path, outfit strength) live under **Settings → Forge Face Consistency**. Via the API, pass
the same values under `alwayson_scripts` → `Forge Face Consistency`
→ `args` (`[enabled, ref_image_base64, refs_dir, strength, restore,
outfit_image_base64, outfit_strength]` — the two outfit args are new and
optional; older 5-arg calls keep working).

## Requirements

- `inswapper_128.onnx` under `models/insightface/` (the ReActor
  convention), or set its path in Settings.
- `insightface` + `onnxruntime` (installed by `install.py`).
- Optional but recommended: `gfpgan_1.4.onnx` for generative face
  restoration after the swap; without it, a high-frequency detail graft
  is used instead (softer, can leave a visible seam).
- For the below-max SDXL reference path: an IP-Adapter FaceID or InstantID
  ControlNet model in `models/ControlNet/`.
- For the outfit / object reference path: a general IP-Adapter model
  (e.g. `ip-adapter_sdxl.safetensors`) in `models/ControlNet/`
  (see above). Not needed for face-only work.

## GFPGAN — two different files, two different jobs

"GFPGAN" names two unrelated model files in Forge; neither substitutes
for the other:

1. **For this extension** (restores the swapped face, runs on ONNX):
   download `gfpgan_1.4.onnx` (~340 MB) from
   https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
   and save it as `models/insightface/gfpgan_1.4.onnx` — right next to
   `inswapper_128.onnx` (also searched:
   `extensions/forge-face-consistency/models/gfpgan_1.4.onnx`). Restart
   Forge. A swap's infotext then reads
   `FaceConsistency restored by: gfpgan` instead of `detail-graft`.
2. **For Forge's built-in face restoration** ("Restore faces" checkbox
   and the Extras GFPGAN panel, runs the PyTorch model): download
   `GFPGANv1.4.pth` (~349 MB) from the official GFPGAN release,
   https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth,
   and save it in `models/GFPGAN/`. Restart Forge, then pick GFPGAN
   under Settings → Face restoration ("Face restoration model") if you
   want it applied to normal generations. (If the file is missing,
   Forge tries to download it automatically the first time restoration
   runs.)

InsightFace's `buffalo_l` / `inswapper` models are released for
non-commercial research use — the same posture as ReActor and InstantID.

## Tests

Offline (no Forge runtime or GPU needed):

```
python -m pytest extensions/forge-face-consistency/tests/ -q
```
