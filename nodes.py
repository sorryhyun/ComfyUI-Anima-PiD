"""ComfyUI nodes: NVIDIA PiD pixel-diffusion decoder for Anima / Qwen-Image latents.

Two nodes:
  * AnimaPiDLoader  — load a PiD qwenimage checkpoint -> ANIMA_PID socket
  * AnimaPiDDecode  — LATENT (+ PiD model) -> IMAGE, 4x super-resolved

PiD REPLACES VAE Decode: it consumes the (normalized) Qwen latent and emits RGB
pixels directly, upscaling 4x in the same pass (latent_grid*8 -> *4). The gemma
text encoder is not loaded at decode time — the net conditions on a fixed,
pre-baked null caption gemma(chi_prompt+"") bundled with the node (~1.4 MB), so
no ~5GB download and no prompt input. Drop AnimaPiDDecode where VAEDecode was:

    checkpoint -> KSampler -> LATENT ─┐
                                      ├─► AnimaPiDDecode ─► IMAGE (4x) -> SaveImage
    AnimaPiDLoader (PiD .pth) ────────┘

The official 2k->4k 4-step checkpoint auto-downloads from the public nvidia/PiD
repo on first use (select the "(auto-download)" entry in AnimaPiDLoader) into
ComfyUI/models/pid/. To use your own, drop a .pth/.safetensors there and pick it
from the dropdown. Weights are NVIDIA NSCLv1 (non-commercial).
"""

import os
import shutil

import torch

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar

from .pid_core import (
    NULL_CAPTION_FILENAME,
    SR_SCALE,
    VAE_DOWN,
    build_pid_net,
    categorize_load_keys,
    comfy_latent_to_lq,
    count_tiles,
    load_null_caption_embs,
    load_pid_weights,
    pid_decode_latent,
    pid_decode_latent_tiled,
)

# Register a ComfyUI models/pid folder for PiD checkpoints (.pth / .safetensors).
_PID_DIR = os.path.join(folder_paths.models_dir, "pid")
os.makedirs(_PID_DIR, exist_ok=True)
folder_paths.add_model_folder_path("pid", _PID_DIR)

# Faithful null caption gemma(chi_prompt + "") — bundled with the node (~1.4 MB,
# derived data), so gemma is never needed. Regen recipe in README "Provenance".
_NULL_CAPTION_PATH = os.path.join(os.path.dirname(__file__), NULL_CAPTION_FILENAME)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Official 4-step qwenimage 2k->4k checkpoint on the (ungated, public) nvidia/PiD
# repo. Auto-fetched into models/pid/ on first use, flattened + renamed to a flat
# filename so the dropdown contract stays uniform.
_HF_PID_REPO = "nvidia/PiD"
_HF_PID_FILE = "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth"
_OFFICIAL_CKPT = "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step.pth"
# Stable dropdown sentinel for the official auto-download. It is ALWAYS present in
# the list (whether or not the file has been fetched), so a saved workflow that
# selected it stays valid across restarts. The trailing marker is cosmetic — load()
# matches on this exact string, not on the real filename.
_AUTODL_ENTRY = _OFFICIAL_CKPT + " (auto-download)"
# Only these are real checkpoints — everything else under models/pid/ (e.g. the
# HF cache's .gitignore / *.metadata) is filtered out of the dropdown.
_CKPT_EXTS = (".pth", ".safetensors", ".pt", ".ckpt", ".bin")


def _download_official_ckpt() -> str:
    """Fetch the official PiD qwenimage checkpoint into models/pid/ (one-time).

    Downloads into the shared HF hub cache (NOT ``local_dir=models/pid``, which
    would litter ``models/pid/.cache/huggingface/`` with ``.gitignore`` /
    ``*.metadata`` files that then show up in the loader dropdown), then copies the
    blob out to a flat, descriptive filename in ``models/pid/`` so it lists like any
    hand-placed checkpoint. Returns the local path."""
    dest = os.path.join(_PID_DIR, _OFFICIAL_CKPT)
    if os.path.exists(dest):
        return dest
    from huggingface_hub import hf_hub_download

    print(
        f"[AnimaPiD] fetching {_HF_PID_REPO}/{_HF_PID_FILE} -> {dest} (one-time, ~public download).\n"
        f"[AnimaPiD] NOTE: PiD weights are NVIDIA NSCLv1 — non-commercial (research/evaluation) use only."
    )
    downloaded = hf_hub_download(repo_id=_HF_PID_REPO, filename=_HF_PID_FILE)
    shutil.copyfile(downloaded, dest)
    return dest


class AnimaPiDModel:
    """Holder for a loaded PiD net + its compute dtype (ANIMA_PID socket)."""

    def __init__(self, net, dtype):
        self.net = net
        self.dtype = dtype


class AnimaPiDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Only real checkpoints — drops HF-cache cruft (.gitignore / *.metadata)
        # that a prior local_dir download may have left under models/pid/.
        files = [f for f in folder_paths.get_filename_list("pid")
                 if f.lower().endswith(_CKPT_EXTS)]
        # The auto-download sentinel is ALWAYS the first entry, present whether or
        # not the official checkpoint has been fetched. This keeps a saved workflow
        # that selected it valid across restarts — the old behaviour dropped the
        # sentinel once the real file existed, invalidating saved graphs.
        files.insert(0, _AUTODL_ENTRY)
        return {
            "required": {
                "ckpt_name": (files,),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("ANIMA_PID",)
    RETURN_NAMES = ("pid_model",)
    FUNCTION = "load"
    CATEGORY = "Anima/PiD"

    def load(self, ckpt_name, dtype):
        # The auto-download sentinel (and the bare official filename) resolve to
        # the official checkpoint, fetching it on first use. Tolerate any
        # "(auto-download)"-suffixed value so older saved workflows keep working.
        if ckpt_name == _AUTODL_ENTRY or ckpt_name == _OFFICIAL_CKPT or "(auto-download)" in ckpt_name:
            path = _download_official_ckpt()
            ckpt_name = _OFFICIAL_CKPT
        else:
            path = folder_paths.get_full_path("pid", ckpt_name)
        if path is None:
            raise FileNotFoundError(
                f"PiD checkpoint {ckpt_name!r} not found under {_PID_DIR}. "
                f"Download nvidia/PiD checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/"
                f"model_ema_bf16.pth and place it there, or select the auto-download entry."
            )
        dt = _DTYPES[dtype]
        device = mm.get_torch_device()
        net = build_pid_net(device, dt)
        missing, unexpected = load_pid_weights(net, path)
        expected_missing, suspect_missing, unexpected = categorize_load_keys(missing, unexpected)
        if expected_missing:
            print(f"[AnimaPiD] note: {len(expected_missing)} expected lq_proj keys absent (distilled student).")
        # Non-lq missing keys or ANY unexpected key mean NET_KWARGS in pid_core.py
        # doesn't match this checkpoint's architecture — the strict=False load hid it.
        if suspect_missing or unexpected:
            print(
                f"[AnimaPiD] WARNING: checkpoint/architecture mismatch — NET_KWARGS in pid_core.py "
                f"may be wrong for this checkpoint.\n"
                f"  {len(suspect_missing)} unexpected MISSING keys (e.g. {suspect_missing[:3]})\n"
                f"  {len(unexpected)} UNEXPECTED keys (e.g. {unexpected[:3]})\n"
                f"  Decode may produce garbage. Verify NET_KWARGS against the checkpoint."
            )
        print(f"[AnimaPiD] loaded {ckpt_name} as {dtype} on {device}")
        return (AnimaPiDModel(net, dt),)


class AnimaPiDDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("ANIMA_PID",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 8}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Latent degradation level PiD assumes. 0.0 = clean decode; "
                                               "higher lets PiD synthesize/hallucinate more detail."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "tile_latent": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8,
                                        "tooltip": "0 = decode whole image at once (4K output may OOM on <=16GB). "
                                                   ">0 = tile the latent (each tile -> tile*32 px) with feather "
                                                   "blending. 64 -> 2048px tiles."}),
                "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 64, "step": 4,
                                         "tooltip": "Latent-space overlap between tiles (pixels = overlap*32). "
                                                    "Larger = fewer seams, slower."}),
                "compile": ("BOOLEAN", {"default": False,
                                        "tooltip": "Per-block torch.compile of the PiD net: each transformer block "
                                                   "is compiled as its own small graph (faster compile, fewer graph "
                                                   "breaks than whole-net — mirrors Anima Block Compile). First run "
                                                   "per output size is slow (compilation), then fast; with tiling on "
                                                   "all tiles share one size so the blocks compile once."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Anima/PiD"

    _null_cache: dict = {}  # device-str -> cached null caption tensor

    def _null_caption(self, device, dtype):
        """The faithful gemma(chi_prompt+'') null the net conditions on (the distill
        path has no CFG, so it's a single fixed null). Bundled with the node and
        loaded once per device."""
        key = str(device)
        cached = type(self)._null_cache.get(key)
        if cached is not None:
            return cached.to(dtype=dtype)
        if not os.path.exists(_NULL_CAPTION_PATH):
            raise FileNotFoundError(
                f"Bundled null caption missing: {_NULL_CAPTION_PATH}\n"
                f"It ships with the node; if it was deleted, regenerate it per the "
                f"'Provenance' section of the node README."
            )
        cap = load_null_caption_embs(_NULL_CAPTION_PATH, device, dtype)
        type(self)._null_cache[key] = cap
        print(f"[AnimaPiD] loaded null caption {tuple(cap.shape)} from bundled file")
        return cap

    def decode(self, pid_model, latent, steps, sigma, seed, tile_latent, tile_overlap,
               compile=False):
        net = pid_model.net
        dt = pid_model.dtype
        device = mm.get_torch_device()

        cap = self._null_caption(device, dt)

        lq = comfy_latent_to_lq(latent["samples"], device, dt)  # (B,16,h,w) normalized
        lh, lw = lq.shape[-2], lq.shape[-1]
        out_h, out_w = lh * VAE_DOWN * SR_SCALE, lw * VAE_DOWN * SR_SCALE
        print(f"[AnimaPiD] decode latent {lh}x{lw} -> {out_h}x{out_w} ({SR_SCALE}x), "
              f"steps={steps} sigma={sigma} tile={tile_latent or 'off'} compile={compile}")

        use_tiling = bool(tile_latent) and (lh > tile_latent or lw > tile_latent)
        # Drive ComfyUI's node progress bar: one tick per SDE step, summed over
        # tiles. (PiD runs its own sampler loop, so without this the node shows
        # no progress.)
        n_tiles = count_tiles(lq, tile_latent, tile_overlap) if use_tiling else 1
        pbar = ProgressBar(steps * n_tiles)
        step_cb = lambda: pbar.update(1)  # noqa: E731
        if use_tiling:
            px = pid_decode_latent_tiled(
                net, lq, steps=steps, sigma=sigma, seed=seed,
                tile=tile_latent, overlap=tile_overlap, dtype=dt, compile=compile,
                caption_embs=cap, step_cb=step_cb,
            )
        else:
            px = pid_decode_latent(net, lq, steps=steps, sigma=sigma, seed=seed,
                                   dtype=dt, compile=compile, caption_embs=cap, step_cb=step_cb)

        # (B,3,H,W) in [-1,1] -> ComfyUI IMAGE (B,H,W,3) in [0,1]
        img = ((px.float() + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu()
        return (img,)


NODE_CLASS_MAPPINGS = {
    "AnimaPiDLoader": AnimaPiDLoader,
    "AnimaPiDDecode": AnimaPiDDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPiDLoader": "Anima PiD Loader",
    "AnimaPiDDecode": "Anima PiD Decode (4x SR)",
}
