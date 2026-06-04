"""Self-contained PiD-Qwen 4-step pixel-diffusion decoder core.

No hydra / imaginaire / gemma. Everything here is derived from the live
nv-tlabs PiD `qwenimage` 2kto4k checkpoint config (captured by introspection):

  * NET_KWARGS         — exact PidNet constructor args
  * STUDENT_T_LIST etc — the distilled 4-step SDE schedule
  * gemma is not loaded at decode time. The distill student uses no CFG, so there
    is no negative branch to encode — the net just needs a *fixed* null caption.
    The faithful null is `gemma(chi_prompt + "")` (== the model's
    `_encode_text_raw([""])`): the qwenimage checkpoint was distilled with a long
    `chi_prompt` instruction prefixed to every caption, so an all-zero y is
    off-distribution. That null is pre-baked and bundled with the node (~1.4MB;
    regen recipe in README "Provenance"), so gemma is never downloaded. The decode
    fns take a `caption_embs` arg; passing None falls back to a zero caption
    (NaN-safe — the y_embedder is `Linear(2304→D, bias=True)` then RMSNorm — but
    off-distribution; the node always passes the faithful null).

The PiD net consumes a *normalized* Qwen latent (LQ_latent) and emits RGB pixels
directly — there is NO VAE decode at the end, and the Qwen VAE is not needed at
all (we feed the latent straight in). Output spatial size = latent_grid * 8 *
sr_scale(=4).
"""

from __future__ import annotations

import os
import shutil
import sys

import torch

from .pid_net import PidNet

# ---- Exact PidNet constructor kwargs (introspected from the live qwenimage ckpt) ----
NET_KWARGS = dict(
    in_channels=3, num_groups=24, hidden_size=1536, pixel_hidden_size=16,
    pixel_attn_hidden_size=1152, pixel_num_groups=16, patch_depth=14, pixel_depth=2,
    num_text_blocks=4, patch_size=16, txt_embed_dim=2304, txt_max_length=300,
    use_text_rope=True, text_rope_theta=10000.0, rope_mode="ntk_aware",
    rope_ref_h=1024, rope_ref_w=1024, repa_encoder_index=6, enable_ed=False,
    ed_compress_ratio=1, ed_depth_per_stage=1, ed_window_size=2, ed_num_heads=None,
    ed_hidden_size=None, ed_use_token_shuffle=True, lq_inject_mode="controlnet",
    lq_in_channels=0, lq_latent_channels=16, lq_hidden_dim=512, lq_num_res_blocks=4,
    lq_gate_type="sigma_aware_per_token_per_dim", lq_interval=2, zero_init_lq=True,
    train_lq_proj_only=False, sr_scale=4, latent_spatial_down_factor=8,
    pit_lq_inject=False, pit_lq_gate_type="sigma_aware_per_token_per_dim",
)

SR_SCALE = 4
VAE_DOWN = 8           # latent grid -> vae-native pixels
MODEL_MAX_LENGTH = 300
CAPTION_CHANNELS = 2304

# Cached faithful null (gemma(chi_prompt + "")) — bundled blob; regen recipe in README.
# A (1, MODEL_MAX_LENGTH, CAPTION_CHANNELS) tensor — the same y the official student
# sees for an empty user prompt. Lives alongside the ckpt in models/pid/.
NULL_CAPTION_FILENAME = "pid_null_caption_gemma.safetensors"
NULL_CAPTION_KEY = "null_caption_embs"
FM_TIMESCALE = 1000.0
STUDENT_T_LIST = [0.999, 0.866, 0.634, 0.342, 0.0]  # 4-step SDE schedule
STUDENT_SAMPLE_STEPS = 4

# Qwen-Image VAE per-channel latent normalization (== ComfyUI Wan21 format,
# scale_factor 1.0; == anima_lora qwen_vae.latents_mean/std).
QWEN_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
QWEN_LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def build_pid_net(device="cuda", dtype=torch.bfloat16) -> PidNet:
    net = PidNet(**NET_KWARGS)
    net = net.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return net


def load_pid_weights(net: PidNet, ckpt_path: str):
    """Load consolidated PiD checkpoint. The official `model_ema_bf16.pth` stores
    keys under a `net.` prefix (PixelDiTModel.state_dict(prefix='net.')); strip it.
    Also tolerates a bare-key state dict and a {'state_dict'/'model': ...} wrapper.

    Returns (missing, unexpected) where `missing` is split-categorized by the caller.
    The load is necessarily `strict=False` (the official student legitimately omits
    `lq_proj` keys — see PidDistillModel.load_state_dict), but that also means a wrong
    NET_KWARGS would load silently. `categorize_load_keys` lets the loader surface a
    real arch/kwargs mismatch (any non-`lq_proj` missing, or *any* unexpected key)
    loudly instead of hiding it behind the expected lq_proj omissions."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("net.") for k in sd):
        sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    return missing, unexpected


def categorize_load_keys(missing, unexpected):
    """Split load_state_dict results into (expected_missing, suspect_missing, unexpected).

    `lq_proj` keys are EXPECTED to be missing for the distilled student. Anything else
    missing — or any unexpected key at all — signals NET_KWARGS doesn't match the
    checkpoint architecture and should be treated as an error, not a note."""
    expected_missing = [k for k in missing if "lq_proj" in k]
    suspect_missing = [k for k in missing if "lq_proj" not in k]
    return expected_missing, suspect_missing, list(unexpected)


def load_null_caption_embs(path: str, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Load the bundled `gemma(chi_prompt + "")` null caption (regen recipe in
    README "Provenance"). Returns (1, MODEL_MAX_LENGTH, CAPTION_CHANNELS)."""
    from safetensors.torch import load_file

    sd = load_file(path)
    if NULL_CAPTION_KEY not in sd:
        raise KeyError(
            f"{path} has no '{NULL_CAPTION_KEY}' tensor (keys: {list(sd)}). "
            f"Regenerate per README 'Provenance'."
        )
    emb = sd[NULL_CAPTION_KEY]
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    exp = (MODEL_MAX_LENGTH, CAPTION_CHANNELS)
    if tuple(emb.shape[-2:]) != exp:
        raise ValueError(
            f"null caption shape {tuple(emb.shape)} != expected (*, {exp[0]}, {exp[1]}). "
            f"Regenerate per README 'Provenance'."
        )
    return emb.to(device=device, dtype=dtype)


def comfy_latent_to_lq(samples: torch.Tensor, device, dtype=torch.bfloat16) -> torch.Tensor:
    """ComfyUI LATENT['samples'] (raw Qwen VAE latent, 4D or 5D) -> PiD LQ_latent
    (per-channel normalized (mu-mean)/std), 4D (B,16,h,w)."""
    x = samples
    if x.ndim == 5:               # (B,C,T,h,w) -> drop singleton frame
        x = x[:, :, 0]
    mean = torch.tensor(QWEN_LATENTS_MEAN, device=x.device, dtype=torch.float32).view(1, 16, 1, 1)
    std = torch.tensor(QWEN_LATENTS_STD, device=x.device, dtype=torch.float32).view(1, 16, 1, 1)
    lq = (x.float() - mean) / std
    return lq.to(device=device, dtype=dtype)


# Nets whose transformer blocks have been compiled in place (idempotent guard).
_COMPILED_NETS: set = set()

# Tri-state cache: None = not yet probed, True/False = host C++ compiler present.
_COMPILE_OK = None


def _host_cpp_compiler() -> str | None:
    """Return the host C++ compiler torch.compile's inductor backend would invoke,
    or None if none is on PATH. Mirrors inductor's own lookup so we can decide BEFORE
    compiling whether the build can possibly succeed.

    inductor codegens a C++ wrapper/kernel and shells out to a compiler: ``cl`` on
    Windows (MSVC), g++/clang++ elsewhere. The build is lazy — it happens at the
    first *call* of a compiled graph, deep in the sample loop — so a missing compiler
    surfaces as ``Compiler: cl is not found`` mid-decode and kills the run. A trivial
    up-front probe is unreliable (a pointwise op can run via Triton alone, never
    touching the C++ wrapper, so it passes while the real blocks still fail), so we
    check for the compiler executable directly instead."""
    for var in ("CXX", "CC"):  # honor explicit overrides, like inductor does
        c = os.environ.get(var)
        if c and shutil.which(c):
            return c
    candidates = ("cl",) if sys.platform == "win32" else ("g++", "c++", "clang++", "gcc", "cc")
    for c in candidates:
        if shutil.which(c):
            return c
    return None


def _compiler_available(device, dtype) -> bool:
    """Whether torch.compile can build on this machine (cached). If no host C++
    compiler is found, warn once and return False so callers fall back to eager
    instead of crashing mid-decode with ``Compiler: cl is not found``."""
    global _COMPILE_OK
    if _COMPILE_OK is not None:
        return _COMPILE_OK
    cc = _host_cpp_compiler()
    if cc is None:
        print(
            "[AnimaPiD] no host C++ compiler found — torch.compile would crash this "
            "decode (inductor needs one to build kernels). Decoding eagerly; the "
            "'compile' toggle is a no-op on this machine.\n"
            "[AnimaPiD] Windows: install MSVC Build Tools and run ComfyUI from a "
            "'x64 Native Tools Command Prompt' (so 'cl.exe' is on PATH) to enable "
            "compile; otherwise just leave the Decode node's 'compile' off."
        )
        _COMPILE_OK = False
    else:
        # Compiler present, but guard against any *other* inductor build failure
        # (toolchain mismatch, missing headers, …) degrading to eager rather than
        # raising. suppress_errors makes dynamo run the original eager bytecode on a
        # backend compile error instead of propagating it.
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except Exception:  # noqa: BLE001 — best-effort; absence just means no net change
            pass
        _COMPILE_OK = True
    return _COMPILE_OK


def _compile_blocks_inplace(net: PidNet) -> None:
    """Per-block `torch.compile`, mirroring anima_lora's `DiT.compile_blocks` and
    the AnimaBlockCompile ComfyUI node: compile each transformer block as its own
    small graph rather than tracing the whole net in one frame.

    PiD's `patch_blocks` (14x, identical) and `pixel_blocks` (2x, identical) each
    collapse to a single graph reused across the stack — far faster to compile and
    far less likely to graph-break or silently fall back to eager than whole-net
    compile. RoPE / positional info is passed INTO each block as args (computed at
    net level), so the blocks compile cleanly without the `precompute_positional_
    caches` dance the whole-net path needed, and the eager net-level forward keeps
    building those caches as normal.

    Idempotent (compiles once per net). Output-resolution changes are handled by
    dynamo's own per-shape specialization — a new size recompiles automatically;
    tiled decode keeps every tile the same size, so the blocks compile just once."""
    if id(net) in _COMPILED_NETS:
        return
    for blocks in (net.patch_blocks, net.pixel_blocks):
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i], mode="default", dynamic=False)
    _COMPILED_NETS.add(id(net))


def get_runner(net: PidNet, dtype, enable: bool) -> PidNet:
    """Return the net to call in the sample loop. With `enable` AND a working
    inductor backend on this machine, the net's transformer blocks are
    torch.compiled in place, once, per-block (see `_compile_blocks_inplace`) and the
    same (now-compiled) net is returned; otherwise the eager net is returned
    unchanged. The `_compiler_available` probe guards against machines without a host
    C compiler (e.g. Windows lacking `cl.exe`), where compiling would crash the
    decode at first block execution rather than at compile time."""
    if enable:
        device = next(net.parameters()).device
        if _compiler_available(device, dtype):
            _compile_blocks_inplace(net)
    return net


def _t_list(num_steps: int, device) -> torch.Tensor:
    full = torch.tensor(STUDENT_T_LIST, device=device, dtype=torch.float32)
    if num_steps == STUDENT_SAMPLE_STEPS:
        return full
    idx = torch.linspace(0, len(full) - 1, num_steps + 1).round().long()
    return full[idx]


@torch.no_grad()
def pid_decode_latent(net: PidNet, lq_latent: torch.Tensor, *, steps: int = 4,
                      sigma: float = 0.0, seed: int = 0, dtype=torch.bfloat16,
                      compile: bool = False, caption_embs: torch.Tensor = None,
                      _runner=None, step_cb=None) -> torch.Tensor:
    """Run the 4-step SDE student. lq_latent: (B,16,h,w) normalized.
    Returns pixels (B,3,H,W) in [-1,1] with H=h*8*4, W=w*8*4.

    `caption_embs` is the fixed null caption y the net conditions on. Pass None for
    the zeros null (off-distribution but NaN-safe); pass a (1 or B, 300, 2304) tensor
    (e.g. from `load_null_caption_embs`) for the faithful gemma(chi_prompt + "") null.
    `compile=True` torch.compiles the net (cached per output resolution; first
    call per (H,W) is slow). `_runner` lets callers (e.g. the tiled path) pass a
    pre-built compiled net so all same-size tiles share one graph. `step_cb`, if
    given, is called once per completed SDE step (drives a host progress bar)."""
    device = lq_latent.device
    B, _, lh, lw = lq_latent.shape
    H, W = lh * VAE_DOWN * SR_SCALE, lw * VAE_DOWN * SR_SCALE
    run = _runner if _runner is not None else get_runner(net, dtype, compile)

    if caption_embs is None:
        cap = torch.zeros(B, MODEL_MAX_LENGTH, CAPTION_CHANNELS, device=device, dtype=dtype)
    else:
        cap = caption_embs.to(device=device, dtype=dtype)
        if cap.shape[0] == 1 and B > 1:
            cap = cap.expand(B, -1, -1)
    deg = torch.full((B,), float(sigma), device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(B, 3, H, W, device=device, dtype=torch.float32, generator=gen)

    tl = _t_list(steps, device)
    autocast = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.autocast("cpu", dtype=dtype)
    with autocast:
        for t_cur, t_next in zip(tl[:-1], tl[1:]):
            t_scaled = t_cur.expand(B) * FM_TIMESCALE
            v = run(x.to(dtype), t_scaled, cap, lq_latent=lq_latent, degrade_sigma=deg)
            s = [B] + [1] * (x.ndim - 1)
            t_c = t_cur.double().view(*s)
            x0 = (x.double() - t_c * v.double())  # velocity -> x0
            if t_next.item() > 0:
                eps = torch.randn(x0.shape, device=device, dtype=torch.float64, generator=gen)
                t_n = t_next.double().view(1).expand(s)
                x = ((1.0 - t_n) * x0 + t_n * eps).float()
            else:
                x = x0.float()
            if step_cb is not None:
                step_cb()
    return x.clamp(-1, 1)


def _tile_positions(dim: int, tile: int, stride: int):
    """Start indices so every tile is exactly `tile` wide (last snapped to edge)."""
    if dim <= tile:
        return [0], dim
    pos = list(range(0, dim - tile + 1, stride))
    if pos[-1] != dim - tile:
        pos.append(dim - tile)
    return pos, tile


def count_tiles(lq_latent: torch.Tensor, tile: int, overlap: int) -> int:
    """How many tiles `pid_decode_latent_tiled` will process for this latent —
    lets the host size a progress bar (total = steps * count_tiles)."""
    Hh, Ww = lq_latent.shape[-2], lq_latent.shape[-1]
    stride = max(1, tile - overlap)
    ys, _ = _tile_positions(Hh, tile, stride)
    xs, _ = _tile_positions(Ww, tile, stride)
    return len(ys) * len(xs)


def _feather_1d(n: int, overlap_px: int, device) -> torch.Tensor:
    """Linear ramp from a small floor->1 over `overlap_px` at each end, 1 in the
    middle. Floor>0 so single-coverage border pixels normalize cleanly."""
    w = torch.ones(n, device=device, dtype=torch.float32)
    if overlap_px > 0:
        ramp = torch.linspace(1.0 / (overlap_px + 1), 1.0, overlap_px, device=device)
        k = min(overlap_px, n)
        w[:k] = torch.minimum(w[:k], ramp[:k])
        w[-k:] = torch.minimum(w[-k:], ramp.flip(0)[:k])
    return w


@torch.no_grad()
def pid_decode_latent_tiled(net: PidNet, lq_latent: torch.Tensor, *, steps: int = 4,
                            sigma: float = 0.0, seed: int = 0, tile: int = 64,
                            overlap: int = 16, dtype=torch.bfloat16,
                            compile: bool = False, caption_embs: torch.Tensor = None,
                            step_cb=None) -> torch.Tensor:
    """Tiled SR decode for latents larger than `tile` (memory bound). Decodes
    overlapping latent tiles and feather-blends them in pixel space. Output
    (B,3, h*32, w*32) in [-1,1]. `step_cb` ticks once per SDE step across all
    tiles (total ticks = steps * count_tiles())."""
    device = lq_latent.device
    B, _, Hh, Ww = lq_latent.shape
    up = VAE_DOWN * SR_SCALE  # 32
    Hout, Wout = Hh * up, Ww * up
    stride = max(1, tile - overlap)
    ys, th = _tile_positions(Hh, tile, stride)
    xs, tw = _tile_positions(Ww, tile, stride)
    ov_px = overlap * up

    acc = torch.zeros(B, 3, Hout, Wout, device=device, dtype=torch.float32)
    wsum = torch.zeros(1, 1, Hout, Wout, device=device, dtype=torch.float32)
    wy = _feather_1d(th * up, ov_px, device)
    wx = _feather_1d(tw * up, ov_px, device)
    wmask = (wy[:, None] * wx[None, :])[None, None]  # (1,1,th*32,tw*32)

    # All tiles share one fixed output size -> blocks compile once, reuse for every tile.
    runner = get_runner(net, dtype, compile)

    n = 0
    for yi in ys:
        for xi in xs:
            tile_lq = lq_latent[..., yi:yi + th, xi:xi + tw]
            px = pid_decode_latent(net, tile_lq, steps=steps, sigma=sigma, seed=seed + n,
                                   dtype=dtype, caption_embs=caption_embs,
                                   _runner=runner, step_cb=step_cb)
            py, px_ = yi * up, xi * up
            acc[..., py:py + th * up, px_:px_ + tw * up] += px.float() * wmask
            wsum[..., py:py + th * up, px_:px_ + tw * up] += wmask
            n += 1
    return (acc / wsum.clamp(min=1e-6)).clamp(-1, 1)
