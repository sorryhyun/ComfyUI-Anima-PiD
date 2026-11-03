"""ComfyUI nodes: NVIDIA PiD pixel-diffusion decoder for Anima / Qwen-Image and
MOSS-Image / FLUX.2 latents.

Two nodes:
  * AnimaPiDLoader  — load a PiD qwenimage or flux2 checkpoint -> ANIMA_PID socket
  * AnimaPiDDecode  — LATENT (+ PiD model) -> IMAGE, 4x super-resolved

PiD REPLACES VAE Decode: it consumes the (normalized) latent and emits RGB pixels
directly, upscaling 4x in the same pass (latent_grid*8 -> *4 for Qwen,
latent_grid*16 -> *4 for FLUX.2). The gemma
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
import comfy.model_patcher
import folder_paths
from comfy.utils import ProgressBar

from .pid_core import (
    COLOR_CALIB_FILENAME,
    NET_KWARGS_V1,
    NULL_CAPTION_FILENAME,
    SR_SCALE,
    apply_color_calib,
    build_pid_net,
    categorize_load_keys,
    comfy_latent_to_lq,
    count_tiles,
    latent_family,
    latent_to_pixel_factor,
    load_color_calib,
    load_null_caption_embs,
    load_pid_weights,
    net_kwargs_for_state_dict,
    pid_decode_latent,
    pid_decode_latent_tiled,
    read_pid_state_dict,
)
from .pid_net.attention_backend import set_attention_backend

# Register a ComfyUI models/pid folder for PiD checkpoints (.pth / .safetensors).
# A `pid:` entry in extra_model_paths.yaml is registered before this runs, so it
# comes first in folder_paths and takes the auto-downloads (see _download_dir).
_PID_DIR = os.path.join(folder_paths.models_dir, "pid")
os.makedirs(_PID_DIR, exist_ok=True)
folder_paths.add_model_folder_path("pid", _PID_DIR)


def _download_dir() -> str:
    """First registered `pid` folder: an extra_model_paths.yaml entry if there is
    one, else models/pid."""
    paths = folder_paths.get_folder_paths("pid")
    return paths[0] if paths else _PID_DIR

# Faithful null caption gemma(chi_prompt + "") — bundled with the node (~1.4 MB,
# derived data), so gemma is never needed. Regen recipe in README "Provenance".
_NULL_CAPTION_PATH = os.path.join(os.path.dirname(__file__), NULL_CAPTION_FILENAME)

# PiD->native-VAE color-match transform — bundled with the node (~0.5 KB). It was
# fitted against the v1 checkpoint, which color-drifts (flat + desaturated) vs the
# native Qwen VAE. Derived by anima_lora bench/pid/fit_color_calib.py.
#
# It is OFF by default since v1.5: upstream fixed colour accuracy in the v1.5
# weights, so stacking the v1-fitted correction on top would over-correct. Left as
# an opt-in toggle for anyone still running a hand-placed v1 checkpoint.
_COLOR_CALIB_PATH = os.path.join(os.path.dirname(__file__), COLOR_CALIB_FILENAME)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Official PiD v1.5 4-step checkpoints on the (ungated, public) nvidia/PiD repo.
# Auto-fetched into models/pid/ on first use, flattened + renamed to a flat
# filename so the dropdown contract stays uniform.
#
# NB the pre-v1.5 checkpoint this node used to fetch now lives under
# checkpoints_deprecated/ upstream — the old path 404s, hence the pins below.
_HF_PID_REPO = "nvidia/PiD"
_OFFICIAL = {
    "qwenimage": "PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step",
    "flux2": "PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step",
}
# Stable dropdown sentinels for the official auto-downloads. They are ALWAYS present
# in the list (whether or not the file has been fetched), so a saved workflow that
# selected one stays valid across restarts. The bare "(auto-download)" is the
# qwenimage checkpoint (what workflows saved before flux2 support selected).
_AUTODL_ENTRIES = {
    "(auto-download)": "qwenimage",
    "(auto-download) flux2 / MOSS-Image": "flux2",
}
# Only these are real checkpoints — everything else under models/pid/ (e.g. the
# HF cache's .gitignore / *.metadata) is filtered out of the dropdown.
_CKPT_EXTS = (".pth", ".safetensors", ".pt", ".ckpt", ".bin")


def _official_ckpt_name(family: str) -> str:
    return _OFFICIAL[family] + ".pth"


def _download_official_ckpt(family: str) -> str:
    """Fetch an official PiD v1.5 checkpoint into models/pid/ (one-time).

    Downloads into the shared HF hub cache (NOT ``local_dir=models/pid``, which
    would litter ``models/pid/.cache/huggingface/`` with ``.gitignore`` /
    ``*.metadata`` files that then show up in the loader dropdown), then copies the
    blob out to a flat, descriptive filename in ``models/pid/`` so it lists like any
    hand-placed checkpoint. Any registered `pid` folder already holding it counts.
    Returns the local path."""
    name = _official_ckpt_name(family)
    found = folder_paths.get_full_path("pid", name)
    if found is not None:
        return found
    dest = os.path.join(_download_dir(), name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    from huggingface_hub import hf_hub_download

    hf_file = f"checkpoints/{_OFFICIAL[family]}/model_ema_bf16.pth"
    print(
        f"[AnimaPiD] fetching {_HF_PID_REPO}/{hf_file} -> {dest} (one-time, ~2.8 GB public download).\n"
        f"[AnimaPiD] NOTE: PiD weights are NVIDIA NSCLv1 — non-commercial (research/evaluation) use only."
    )
    downloaded = hf_hub_download(repo_id=_HF_PID_REPO, filename=hf_file)
    shutil.copyfile(downloaded, dest)
    return dest


class AnimaPiDModel:
    """Holder for a ModelPatcher-wrapped PiD net + its compute dtype (ANIMA_PID socket).

    The net is wrapped in a comfy ModelPatcher so ComfyUI's model manager owns its
    VRAM residency exactly like a UNET/VAE: it rests on the offload (CPU) device and
    is pulled onto the GPU — evicting other models as needed — via load_models_gpu at
    decode time, then evicted on-demand when something else needs the VRAM. `.net` is
    the underlying module (what the pid_core decode fns consume). `family` is the
    latent family the checkpoint decodes ("qwenimage" or "flux2")."""

    def __init__(self, patcher, dtype, family="qwenimage"):
        self.patcher = patcher
        self.dtype = dtype
        self.family = family

    @property
    def net(self):
        return self.patcher.model


class AnimaPiDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Only real checkpoints — drops HF-cache cruft (.gitignore / *.metadata)
        # left by a prior local_dir download, and the null-caption blob (a
        # .safetensors, but not a checkpoint) if a copy was placed under models/pid/.
        files = [f for f in folder_paths.get_filename_list("pid")
                 if f.lower().endswith(_CKPT_EXTS)
                 and os.path.basename(f) not in (NULL_CAPTION_FILENAME, COLOR_CALIB_FILENAME)]
        # The auto-download sentinels are ALWAYS the first entries, present whether
        # or not the official checkpoint has been fetched. This keeps a saved
        # workflow that selected one valid across restarts.
        files = list(_AUTODL_ENTRIES) + files
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
        # An auto-download sentinel resolves to its official checkpoint, fetching it
        # on first use. Any other "(auto-download)"-suffixed value (older saved
        # workflows embedded the filename) maps to qwenimage.
        family = _AUTODL_ENTRIES.get(ckpt_name)
        if family is None and "(auto-download)" in ckpt_name:
            family = "flux2" if "flux2" in ckpt_name else "qwenimage"
        if family is not None:
            path = _download_official_ckpt(family)
            ckpt_name = _official_ckpt_name(family)
        else:
            path = folder_paths.get_full_path("pid", ckpt_name)
        if path is None:
            raise FileNotFoundError(
                f"PiD checkpoint {ckpt_name!r} not found under {folder_paths.get_folder_paths('pid')}. "
                f"Download it from nvidia/PiD and place it there, "
                f"or select an auto-download entry."
            )
        dt = _DTYPES[dtype]
        # Build the net on the OFFLOAD (CPU) device, not the GPU — the same load/
        # offload split every core ComfyUI loader uses (UNETLoader builds on the
        # offload device and hands a ModelPatcher to the manager; see comfy.sd.
        # load_diffusion_model_state_dict). The patcher (below) then lets ComfyUI's
        # model manager own residency: it stays on CPU until decode pulls it onto
        # the GPU via load_models_gpu, and is evicted on-demand when something else
        # (e.g. the next KSampler run) needs the VRAM.
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        # Architecture comes from the checkpoint, not from the filename — v1 and v1.5
        # differ (gate shape, LQ trunk width, PiT injection), and a user may still have
        # a hand-placed v1 file. Read the state dict first, then build to match it.
        sd = read_pid_state_dict(path)
        net_kwargs = net_kwargs_for_state_dict(sd)
        arch = "v1" if net_kwargs is NET_KWARGS_V1 else "v1.5"
        family = latent_family(net_kwargs)
        net = build_pid_net(offload_device, dt, net_kwargs)
        missing, unexpected = load_pid_weights(net, sd)
        expected_missing, suspect_missing, unexpected = categorize_load_keys(missing, unexpected)
        if expected_missing:
            print(f"[AnimaPiD] note: {len(expected_missing)} expected lq_proj keys absent (distilled student).")
        # Non-lq missing keys or ANY unexpected key mean the net kwargs in pid_core.py
        # don't match this checkpoint's architecture — the strict=False load hid it.
        if suspect_missing or unexpected:
            print(
                f"[AnimaPiD] WARNING: checkpoint/architecture mismatch — the selected net kwargs "
                f"in pid_core.py may be wrong for this checkpoint.\n"
                f"  {len(suspect_missing)} unexpected MISSING keys (e.g. {suspect_missing[:3]})\n"
                f"  {len(unexpected)} UNEXPECTED keys (e.g. {unexpected[:3]})\n"
                f"  Decode may produce garbage. Verify the net kwargs against the checkpoint."
            )
        # Hand the net to ComfyUI's model manager via a ModelPatcher (no patches —
        # we only want its load/offload lifecycle, not weight patching). With no
        # comfy-cast modules the manager simply full-loads it onto the GPU and
        # evicts it whole when VRAM is needed, which is exactly the residency
        # convention we want for a decode-time model.
        patcher = comfy.model_patcher.ModelPatcher(net, load_device=load_device, offload_device=offload_device)
        print(
            f"[AnimaPiD] loaded {ckpt_name} as {dtype} "
            f"(arch={arch}, latent={family}, offload={offload_device}, load={load_device}; ComfyUI-managed)"
        )
        return (AnimaPiDModel(patcher, dt, family),)


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
                                                   ">0 = tile with feather blending; each tile is tile*32 output "
                                                   "px for either latent family (64 -> 2048px tiles; a FLUX.2 / "
                                                   "MOSS latent tiles at half this many cells)."}),
                "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 64, "step": 4,
                                         "tooltip": "Overlap between tiles, in the same units (pixels = overlap*32). "
                                                    "Larger = fewer seams, slower."}),
                "compile": ("BOOLEAN", {"default": False,
                                        "tooltip": "Per-block torch.compile of the PiD net: each transformer block "
                                                   "is compiled as its own small graph (faster compile, fewer graph "
                                                   "breaks than whole-net — mirrors Anima Block Compile). First run "
                                                   "per output size is slow (compilation), then fast; with tiling on "
                                                   "all tiles share one size so the blocks compile once."}),
                "use_calib": ("BOOLEAN", {"default": False,
                                          "tooltip": "Apply the bundled PiD->native-VAE color-match transform after "
                                                     "decode. It was fitted against the pre-v1.5 qwenimage checkpoint, "
                                                     "which decoded flat and desaturated vs the native Qwen VAE. v1.5 "
                                                     "fixes colour accuracy upstream, so this is OFF by default — "
                                                     "applying it on v1.5 would over-correct. Turn on only for a "
                                                     "hand-placed v1 checkpoint."}),
                "attention": (["auto", "sdpa", "flash", "sage"], {"default": "auto",
                                          "tooltip": "Attention backend for the PiD net. 'auto' honors ComfyUI's "
                                                     "--use-sage-attention / --use-flash-attention launch flags (sage "
                                                     "wins), else plain SDPA. 'sdpa' already runs PyTorch's built-in "
                                                     "flash kernel here, so 'flash' is ~identical; 'sage' (INT8) is the "
                                                     "real speedup. Masked/fp32/CPU paths always fall back to SDPA."}),
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

    _calib_cache: dict = {}  # device-str -> cached (M, b) color-match transform

    def _color_calib(self, device):
        """The bundled PiD->native-VAE color-match transform (M, b), loaded once per
        device. Corrects the qwenimage checkpoint's flat/desaturated drift."""
        key = str(device)
        cached = type(self)._calib_cache.get(key)
        if cached is not None:
            return cached
        if not os.path.exists(_COLOR_CALIB_PATH):
            raise FileNotFoundError(
                f"Bundled color calib missing: {_COLOR_CALIB_PATH}\n"
                f"It ships with the node; regenerate it with anima_lora "
                f"bench/pid/fit_color_calib.py, or set use_calib=False."
            )
        M, b = load_color_calib(_COLOR_CALIB_PATH, device)
        type(self)._calib_cache[key] = (M, b)
        print("[AnimaPiD] loaded color-match calib (3x3 + bias) from bundled file")
        return (M, b)

    def decode(self, pid_model, latent, steps, sigma, seed, tile_latent, tile_overlap,
               compile=False, use_calib=False, attention="auto"):
        patcher = pid_model.patcher
        dt = pid_model.dtype
        load_device = patcher.load_device

        # Hand residency to ComfyUI's model manager: this evicts the KSampler DiT
        # (and anything else) as needed and brings the PiD net onto the GPU, then
        # leaves it loaded so the manager can evict it on-demand when the next run
        # needs the VRAM. No manual offload — the manager owns the lifecycle, so a
        # net kept resident across decodes also keeps its compiled blocks warm.
        mm.load_models_gpu([patcher])
        net = patcher.model

        # "auto" -> None defers to the launch flags; otherwise force the backend.
        set_attention_backend(None if attention == "auto" else attention)

        cap = self._null_caption(load_device, dt)

        family = getattr(pid_model, "family", "qwenimage")
        lq = comfy_latent_to_lq(latent["samples"], load_device, dt,
                                latent_channels=net.lq_proj.latent_channels)
        lh, lw = lq.shape[-2], lq.shape[-1]
        up = latent_to_pixel_factor(net)
        out_h, out_w = lh * up, lw * up
        # tile_latent / tile_overlap count 32-px units (one Qwen latent cell), so a
        # setting keeps its pixel size across latent families.
        tile = tile_latent * 32 // up
        overlap = tile_overlap * 32 // up
        print(f"[AnimaPiD] decode {family} latent {lh}x{lw} -> {out_h}x{out_w} ({SR_SCALE}x), "
              f"steps={steps} sigma={sigma} tile={f'{tile} cells' if tile else 'off'} compile={compile}")

        use_tiling = bool(tile) and (lh > tile or lw > tile)
        # Drive ComfyUI's node progress bar: one tick per SDE step, summed over
        # tiles. (PiD runs its own sampler loop, so without this the node shows
        # no progress.)
        n_tiles = count_tiles(lq, tile, overlap) if use_tiling else 1
        pbar = ProgressBar(steps * n_tiles)
        step_cb = lambda: pbar.update(1)  # noqa: E731
        if use_tiling:
            # global_lq defaults True: the LQ projection runs once over the full
            # latent so its GroupNorm sees whole-image statistics, keeping the tiled
            # tone aligned with the whole-image decode. The bundled color calib is
            # fit through this same path (see metadata global_lq/tile_latent).
            px = pid_decode_latent_tiled(
                net, lq, steps=steps, sigma=sigma, seed=seed,
                tile=tile, overlap=overlap, dtype=dt, compile=compile,
                caption_embs=cap, step_cb=step_cb,
            )
        else:
            px = pid_decode_latent(net, lq, steps=steps, sigma=sigma, seed=seed,
                                   dtype=dt, compile=compile, caption_embs=cap, step_cb=step_cb)

        # (B,3,H,W) in [-1,1] -> ComfyUI IMAGE (B,H,W,3) in [0,1]
        img = ((px.float() + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).contiguous()
        if use_calib and family != "qwenimage":
            print("[AnimaPiD] use_calib ignored: the bundled calib was fit on the Qwen-Image checkpoint.")
        elif use_calib:
            M, b = self._color_calib(load_device)
            img = apply_color_calib(img, M, b)  # PiD->native-VAE color match (channel-last)
        return (img.cpu(),)


NODE_CLASS_MAPPINGS = {
    "AnimaPiDLoader": AnimaPiDLoader,
    "AnimaPiDDecode": AnimaPiDDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPiDLoader": "Anima PiD Loader",
    "AnimaPiDDecode": "Anima PiD Decode (4x SR)",
}
