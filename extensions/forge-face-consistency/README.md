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
`blended-swap` / `swap`) and the strength into the image infotext. The
swap and blended-swap paths also record the ArcFace similarity
before/after, plus a warning when the result lands below the verify
threshold (Settings → Forge Face Consistency). The ControlNet reference
paths form identity during diffusion, so their similarity is recorded as
`n/a (controlnet)` rather than measured. A downgrade (e.g. Flux below
max, or no adapter found) is always recorded — never silent.

## Use

1. Open the **Forge Face Consistency** accordion (txt2img or img2img).
2. Tick Enable, drop in a reference face photo (or point *Reference images
   folder* at a folder of photos of the person — they are averaged into one
   outlier-cleaned identity template). For the ControlNet reference path,
   the folder's image files are tried in sorted order until one decodes.
3. Set strength: max for a guaranteed swap, lower for a guided reference.

Global defaults (enable, strength, restore, verify threshold, inswapper
path) live under **Settings → Forge Face Consistency**. Via the API, pass
the same five values under `alwayson_scripts` → `Forge Face Consistency`
→ `args` (`[enabled, ref_image_base64, refs_dir, strength, restore]`).

## Requirements

- `inswapper_128.onnx` under `models/insightface/` (the ReActor
  convention), or set its path in Settings.
- `insightface` + `onnxruntime` (installed by `install.py`).
- Optional: `gfpgan_1.4.onnx` next to the inswapper for generative face
  restoration; without it, a high-frequency detail graft is used instead.
- For the below-max SDXL reference path: an IP-Adapter FaceID or InstantID
  ControlNet model in `models/ControlNet/`.

InsightFace's `buffalo_l` / `inswapper` models are released for
non-commercial research use — the same posture as ReActor and InstantID.

## Tests

Offline (no Forge runtime or GPU needed):

```
python -m pytest extensions/forge-face-consistency/tests/ -q
```
