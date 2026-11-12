import torch

from ._ops import ops
from .experts import experts_forward, fp8_experts_forward, fused_experts, register
from .vendored import gpt_oss_mlp, yamoe_ref

__all__ = [
    # Debug
    "ops",
    # transformers experts layout (CUDA and ROCm)
    "experts_forward",
    "fp8_experts_forward",
    "fused_experts",
    "register",
    # Vendored reference implementations
    "yamoe_ref",
    "gpt_oss_mlp",
]

# The gpt-oss layer and ops use cuBLAS / CUTLASS and are only built for CUDA.
if torch.version.hip is None:
    from .layers import Yamoe

    gather = ops.gather
    scatter = ops.scatter
    sort = ops.sort
    bincount_cumsum = ops.bincount_cumsum
    batch_mm = ops.batch_mm
    experts = ops.experts
    experts_backward = ops.experts_backward

    __all__ += [
        "Yamoe",
        "gather",
        "scatter",
        "sort",
        "bincount_cumsum",
        "batch_mm",
        "experts",
        "experts_backward",
    ]
