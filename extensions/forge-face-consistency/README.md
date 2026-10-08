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
- Optional but recommended: `gfpgan_1.4.onnx` for generative face
  restoration after the swap; without it, a high-frequency detail graft
  is used instead (softer, can leave a visible seam).
- For the below-max SDXL reference path: an IP-Adapter FaceID or InstantID
  ControlNet model in `models/ControlNet/`.

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
