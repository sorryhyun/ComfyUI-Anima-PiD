# ComfyUI-Anima-PiD

NVIDIA **PiD** (Pixel Diffusion Decoder) as a drop-in replacement for **VAE Decode**
on Anima / Qwen-Image latents and on MOSS-Image / FLUX.2 latents. It takes a `LATENT` and emits a **4× super-resolved
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
for Qwen latents (a 128×128 latent → 4096×4096) and `latent_grid × 16 × 4` for FLUX.2 /
MOSS latents (a 64×64 latent → 4096×4096) — 4× the native VAE decode either way.

## MOSS-Image / FLUX.2

PiD v1.5 ships a FLUX.2 checkpoint
(`PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step`). Pick
**`(auto-download) flux2 / MOSS-Image`** in the loader and wire the MOSS KSampler's
`LATENT` into the decode node. It takes the packed, BatchNorm-normalized
`(B, 128, H/16, W/16)` latent — the MOSS DiT latent and ComfyUI's Flux2 `LATENT` —
as is, and unpatchifies it 2×2 to `(B, 32, H/8, W/8)` inside its LQ projection
(upstream `PID_SR4X_V1PT5_FOR_FLUX2`). It shares the qwenimage checkpoint's prompt
prefix and 4-step schedule, so the same bundled null caption is used. The loader
reads the family from the checkpoint (input width of the first LQ conv: 16 → Qwen,
32 → FLUX.2), and the decode node refuses a latent of the other family.

## Checkpoint: PiD v1.5

The node now fetches the **v1.5** qwenimage checkpoint
(`PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step`, released 2026-07-09).
Upstream's v1.5 changes, all of which matter here:

- **better colour accuracy** — which is why `use_calib` is now **off** by default
  (the bundled calib was fitted against v1's drift; see the node options below),
- **no grid artifacts in the corners** — the LQ projection now uses `replicate`
  conv padding instead of zero padding,
- **trained with more anime data and small-face data**.

Architecturally v1.5 is *not* a drop-in reload: it widens the LQ trunk
(`lq_hidden_dim` 512 → 1024), swaps the injection gate from per-token-per-dim to a
cheaper **per-token scalar** gate, moves the RoPE reference to the 2048 training
resolution, and adds a dedicated **LQ injection into the PiT pixel blocks**. The
loader picks the matching architecture **from the checkpoint itself**, so a
hand-placed v1 file still loads (it is detected by the absence of `lq_proj.pit_head`).

> The pre-v1.5 checkpoint moved to `checkpoints_deprecated/` upstream, so the path
> older versions of this node auto-downloaded now 404s. Update the node.

### Why "4×" when the name says "2k→4k"

`sr4x` is the **scale factor** and `res2kto4k` is the **trained output range** — they
are not the same number. PiD upscales **4× relative to the source latent's native VAE
decode** (output = `latent_grid × 8 × 4`), and the weights are built around that:
the LQ projection's pixel-unshuffle factor is `patch_size // sr_scale = 4`, so the
scale is baked into the tensor shapes and is not a knob. `res2kto4k` says the model
was trained to *emit* 2048–4096px, i.e. for source latents of 512–1024px. A 1024px
Anima generation → 4096px out, which is the intended operating point. Feeding a
2048px generation would ask for 8192px — outside the trained range.

## Install

1. Copy/clone this folder into `ComfyUI/custom_nodes/`.
2. **No manual download needed** — in **Anima PiD Loader**, pick
   `(auto-download)` (Qwen-Image / Anima) or `(auto-download) flux2 / MOSS-Image`
   and the official checkpoint is pulled from the public `nvidia/PiD` repo into
   `ComfyUI/models/pid/` on first run (one-time, ~2.8 GB each). To keep the
   checkpoints on another disk, add a `pid:` entry to `extra_model_paths.yaml`;
   the first registered `pid` folder takes the downloads, and a checkpoint already
   present in any of them is not fetched again. To use your own checkpoint
   instead, drop a `.pth`/`.safetensors` into a `pid` folder and select it from
   the dropdown:
   ```bash
   hf download nvidia/PiD --local-dir /tmp/pid \
     --include "checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/*"
   mkdir -p ComfyUI/models/pid
   cp /tmp/pid/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth \
      ComfyUI/models/pid/pid_qwenimage_2kto4k_4step.pth
   ```
   (The Qwen VAE and gemma are **not** needed at decode time — PiD emits pixels
   directly and uses the bundled fixed null caption.)

## Nodes

- **Anima PiD Loader** — `ckpt_name` (from the `pid` folders), `dtype` → `ANIMA_PID`.
- **Anima PiD Decode (4x SR)** — `ANIMA_PID` + `LATENT` → `IMAGE`.
  - `steps` (default 4) — distilled student steps.
  - `sigma` (default 0.0) — assumed latent degradation; 0 = clean, higher = more
    synthesized detail.
  - `tile_latent` (default 64) — `0` decodes the whole image at once (4K may OOM
    on ≤16 GB); `>0` tiles with feather blending, each tile `tile×32` output px.
    **64 → 2048 px tiles** (~7 GB peak in bf16). The unit is one Qwen latent cell
    (32 output px), so a FLUX.2 / MOSS latent tiles at `tile/2` of its 16-px cells
    and the pixel size stays the same.
  - `tile_overlap` (default 16) — overlap between tiles, same unit (px = `overlap×32`).
  - `compile` (default off) — `torch.compile` the PiD net. ~1.8× faster warm
    (e.g. 3.8s → 2.1s for a 2048px tile), after a one-time ~37s compilation **per
    output resolution**. With tiling on, every tile is the same size so it
    compiles once and all tiles reuse the graph — keep `tile_latent` fixed across
    runs to keep hitting the cache.
  - `use_calib` (default **off**) — apply the bundled **color-match transform**
    (`pid_color_calib.safetensors`) after decode. It was fitted against the
    **pre-v1.5** `qwenimage` checkpoint, which decoded **flat and desaturated** vs
    the native Qwen VAE. **PiD v1.5 fixes colour accuracy upstream**, so the
    transform is off by default — stacking it on v1.5 would over-correct. Turn it
    on only if you are running a hand-placed v1 checkpoint. Ignored for the FLUX.2
    checkpoint. The transform is a
    static linear `out = (rgb @ M.T + b)` fit against native-VAE decodes (contrast
    ≈ ×1.11, saturation ≈ ×1.12, no hue tint). See **Provenance**.

## Latent convention

FLUX.2 / MOSS latents go in unchanged (they are already BatchNorm-normalized).

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

**Bundled color calib** (`pid_color_calib.safetensors`) corrects the **v1**
checkpoint's flat/desaturated drift vs the native Qwen VAE; it is **not** applied to
v1.5 by default, which fixes colour upstream. Keys: `linear_M` (3×3), `linear_b`
(3,); applied as `out = (rgb01 @ linear_M.T + linear_b).clamp(0,1)`. To regenerate
(e.g. after a checkpoint or step-count change — the drift is timestep-dependent),
run the fitter in the `anima_lora` repo:

```bash
# the bundled calib was fit one-(middle)-latent-per-artist over the full dataset:
uv run python bench/pid/fit_color_calib.py --one_per_artist --num_images 0 --steps 4
# -> bench/pid/results/<ts>-per-artist/pid_color_calib.safetensors
```

It decodes cached Anima latents both ways (native Qwen VAE reference vs PiD), then
least-squares fits the linear transform PiD→VAE. The report also prints the
per-image spread: the static transform fixes the *systematic* drift; residual
per-image luminance variance is the 4-step SDE's own "early-termination whitening"
(acknowledged upstream) and is not removable by a static color map — raise `steps`
to shrink it. Fit at the same `--steps` you decode with.
