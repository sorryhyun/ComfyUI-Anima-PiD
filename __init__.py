"""Anima PiD ComfyUI custom nodes.

NVIDIA PiD (Pixel Diffusion Decoder) as a drop-in replacement for VAE Decode on
Anima / Qwen-Image latents: consumes a LATENT and emits a 4x super-resolved
IMAGE in one 4-step pass. The gemma text encoder is never loaded — the distilled
path has no CFG and conditions on a fixed null caption gemma(chi_prompt+"") that
ships pre-baked with the node (~1.4 MB), so there is no multi-GB text-encoder
download and no prompt input.

* ``AnimaPiDLoader`` - load a PiD qwenimage checkpoint -> ``ANIMA_PID`` socket.
* ``AnimaPiDDecode`` - ``ANIMA_PID`` + ``LATENT`` -> ``IMAGE`` (4x), with optional
  latent tiling for 4K on limited VRAM.

The PiD network is vendored self-contained under ``pid_net/`` (Apache-2.0, from
nv-tlabs/PiD) — no hydra / imaginaire dependency. The PiD *weights* are NVIDIA
NSCLv1 (non-commercial) and are supplied by the user, not bundled.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
