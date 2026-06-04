"""Single-GPU stubs for the two pieces of NVIDIA-PiD framework the vendored
network files reference.

The PiD network (`pid_net.py` / `pixeldit_official.py`) only invokes the
context-parallel helpers when `self._cp_group is not None`, which never happens
in a single-process ComfyUI run (CP is a multi-GPU training feature). So the
identity functions below are never actually called on a hot path — they exist
purely so the `import` lines resolve without dragging in the imaginaire
framework. `log` is a plain module logger replacing `imaginaire.utils.log`.
"""

import logging

log = logging.getLogger("comfyui-anima-pid")


def split_inputs_cp(x, seq_dim, cp_group):  # noqa: ARG001 - signature parity only
    return x


def cat_outputs_cp_with_grad(x, seq_dim, cp_group):  # noqa: ARG001
    return x
