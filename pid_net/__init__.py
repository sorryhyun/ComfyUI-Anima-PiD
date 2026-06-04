"""Vendored, self-contained NVIDIA-PiD pixel-diffusion network.

Copied verbatim from nv-tlabs/PiD (Apache-2.0) — `pid/_src/networks/{pid_net,
pixeldit_official,lq_projection_2d}.py` — with only their cross-imports rewritten
to local relative imports + `_stubs` (see `_stubs.py`). No hydra / imaginaire /
omegaconf dependency. Refresh with the node's `vendor-sync` note in README.

NOTE: the PiD *weights* are NVIDIA NSCLv1 (non-commercial); this vendored *code*
is Apache-2.0. The weights are not redistributed here — the user supplies them.
"""

from .pid_net import PidNet

__all__ = ["PidNet"]
