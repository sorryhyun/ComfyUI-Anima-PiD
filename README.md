# ComfyUI-Anima-PiD

NVIDIA **PiD** (Pixel Diffusion Decoder) as a drop-in replacement for **VAE Decode**
on Anima / Qwen-Image latents. It takes a `LATENT` and emits a **4× super-resolved
`IMAGE`** in a single 4-step pass — decode *and* upscale fused into one node.

The gemma text encoder is **never loaded** — no ~5 GB download and no prompt
input. The distilled 4-step path uses no classifier-free guidance, so the net
just needs a *fixed* null caption. We use the faithful one: `gemma(chi_prompt +
"")` — the model's own no-user-prompt null (`_encode_text_raw([""])`), pre-baked
once and **bundled with the node** (`pid_null_caption_gemma.safetensors`, ~1.4 MB,
derived data). See **Provenance** below for how it's generated.

Why the faithful null and not just zeros? The qwenimage student was distilled
with a long `chi_prompt` instruction prefixed to *every* caption, so an all-zero
`y` is off-distribution. An A/B (four 2048px decodes, same seed/latent, zeros vs
faithful null) gave **~29 dB PSNR**: structurally identical, differing only in
fine screentone/line-edge detail — small enough that zeros also "works", but the
bundled null is the in-distribution choice and costs nothing extra. (The
checkpoint architecture is exactly reproduced — a clean load shows zero
unexpected / zero non-`lq_proj` missing keys.)

## Flow

PiD replaces VAE Decode. Drop `AnimaPiDDecode` where `VAEDecode` was:

```
 checkpoint → KSampler → LATENT ─┐
                                 ├─► Anima PiD Decode (4x SR) ─► IMAGE → Save Image
 Anima PiD Loader (PiD .pth) ────┘
```

There is **no second KSampler** and your Anima model does **not** connect to PiD —
PiD runs its own internal 4-step pixel diffusion. Output size = `latent_grid × 8 × 4`
(e.g. a 64×64 latent → 2048×2048; a 128×128 latent → 4096×4096).

## Install

1. Copy/clone this folder into `ComfyUI/custom_nodes/`.
2. **No manual download needed** — in **Anima PiD Loader**, pick the
   `…distill_4step (auto-download)` entry and the official checkpoint is pulled
   from the public `nvidia/PiD` repo into `ComfyUI/models/pid/` on first run
   (one-time). To use your own checkpoint instead, drop a `.pth`/`.safetensors`
   into `ComfyUI/models/pid/` and select it from the dropdown:
   ```bash
   hf download nvidia/PiD --local-dir /tmp/pid \
     --include "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/*"
   mkdir -p ComfyUI/models/pid
   cp /tmp/pid/checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth \
      ComfyUI/models/pid/pid_qwenimage_2kto4k_4step.pth
   ```
   (The Qwen VAE and gemma are **not** needed at decode time — PiD emits pixels
   directly and uses the bundled fixed null caption.)

## Nodes

- **Anima PiD Loader** — `ckpt_name` (from `models/pid/`), `dtype` → `ANIMA_PID`.
- **Anima PiD Decode (4x SR)** — `ANIMA_PID` + `LATENT` → `IMAGE`.
  - `steps` (default 4) — distilled student steps.
  - `sigma` (default 0.0) — assumed latent degradation; 0 = clean, higher = more
    synthesized detail.
  - `tile_latent` (default 64) — `0` decodes the whole image at once (4K may OOM
    on ≤16 GB); `>0` tiles the latent (each tile → `tile×32` px) with feather
    blending. **64 → 2048 px tiles** (~7 GB peak in bf16).
  - `tile_overlap` (default 16) — latent overlap between tiles (px = `overlap×32`).
  - `compile` (default off) — `torch.compile` the PiD net. ~1.8× faster warm
    (e.g. 3.8s → 2.1s for a 2048px tile), after a one-time ~37s compilation **per
    output resolution**. With tiling on, every tile is the same size so it
    compiles once and all tiles reuse the graph — keep `tile_latent` fixed across
    runs to keep hitting the cache.
  - `use_calib` (default **on**) — apply the bundled **color-match transform**
    (`pid_color_calib.safetensors`) after decode. The `qwenimage` PiD checkpoint
    decodes slightly **flat and desaturated** vs the native Qwen VAE (a known PiD
    color drift — fixed upstream only for the `flux2 _2606` checkpoint, not for
    `qwenimage`). The transform is a static linear `out = (rgb @ M.T + b)` fit
    against native-VAE decodes; it corrects the *systematic* drift (contrast
    ≈ ×1.11, saturation ≈ ×1.12, no hue tint). Turn off for raw PiD output. See
    **Provenance**.

## Latent convention

ComfyUI stores raw Qwen/Wan VAE latents; PiD wants the per-channel **normalized**
latent. The node applies `(latent − mean)/std` (the standard Qwen
`latents_mean/std`, `scale_factor=1.0`) internally — the same convention
`anima_lora`'s `encode_pixels_to_latents` produces. If a future Anima latent
format uses a non-1.0 `scale_factor`, update `pid_core.QWEN_LATENTS_*`.

## Licensing

- **This wrapper code**: MIT (`LICENSE`).
- **Vendored PiD network** (`pid_net/`): Apache-2.0, from
  [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD), cross-imports rewritten to be
  self-contained (no hydra/imaginaire). Refresh by re-copying
  `pid/_src/networks/{pid_net,pixeldit_official,lq_projection_2d}.py` and
  re-applying the local-import rewrites in `pid_net/`.
- **PiD weights**: NVIDIA **NSCLv1 — non-commercial only**. Not redistributed
  here; you download them yourself. Do not ship them in a commercial product.
- **Bundled null caption** (`pid_null_caption_gemma.safetensors`): a fixed
  embedding derived from `gemma-2-2b-it`, subject to Google's
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms). It is derived data
  (not gemma weights); regeneration recipe in **Provenance** below.
- **Bundled color calib** (`pid_color_calib.safetensors`, ~0.5 KB): a 3×3 + bias
  linear color transform, fit by measuring this checkpoint's decode against the
  native Qwen VAE. Derived data; regeneration recipe in **Provenance** below.

## Provenance

Net constructor config + the 4-step SDE schedule (`t_list=[0.999, 0.866, 0.634,
0.342, 0.0]`, velocity prediction, timescale 1000) were captured from the live
`qwenimage` 2kto4k checkpoint and baked into `pid_core.py` so no hydra config
resolution is needed at runtime.

**Bundled null caption** (`pid_null_caption_gemma.safetensors`) reproduces the
qwenimage student's `PixelDiTModel._encode_text_raw([""])`, i.e. its no-user-prompt
null. To regenerate (only needed if upstream changes the chi_prompt):

1. Load `gemma-2-2b-it` from `Efficient-Large-Model/gemma-2-2b-it`
   (`AutoModelForCausalLM(...).get_decoder()`, bf16); tokenizer `padding_side="right"`.
2. `chi_prompt_str = "\n".join(CHI_PROMPT)` — the prompt list is in upstream
   `pid/_src/configs/pid/experiment/shared_config.py` (`_CHI_PROMPT`).
3. Tokenize `[chi_prompt_str + ""]` with `padding="max_length"`, `truncation=True`,
   `max_length = len(tok.encode(chi_prompt_str)) + 300 - 2`.
4. `embs = text_encoder(input_ids, attention_mask)[0]`, then select
   `[0] + list(range(-299, 0))` → `(1, 300, 2304)`; save bf16 under key
   `null_caption_embs`.

**Bundled color calib** (`pid_color_calib.safetensors`) corrects this checkpoint's
flat/desaturated drift vs the native Qwen VAE. Keys: `linear_M` (3×3), `linear_b`
(3,); applied as `out = (rgb01 @ linear_M.T + linear_b).clamp(0,1)`. To regenerate
(e.g. after a checkpoint or step-count change — the drift is timestep-dependent),
run the fitter in the `anima_lora` repo:

```bash
uv run python bench/pid/fit_color_calib.py --num_images 24 --steps 4
# -> bench/pid/results/<ts>-colorcalib/pid_color_calib.safetensors
```

It decodes cached Anima latents both ways (native Qwen VAE reference vs PiD), then
least-squares fits the linear transform PiD→VAE. The report also prints the
per-image spread: the static transform fixes the *systematic* drift; residual
per-image luminance variance is the 4-step SDE's own "early-termination whitening"
(acknowledged upstream) and is not removable by a static color map — raise `steps`
to shrink it. Fit at the same `--steps` you decode with.
