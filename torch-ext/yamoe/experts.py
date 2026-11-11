"""Experts forward for the transformers experts layout, on CUDA and ROCm.

`fused_experts` is the functional entry point (gate activation SiLU or GELU tanh). `experts_forward` matches the transformers
`ExpertsInterface` signature, so it can be registered as an experts implementation:

    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    ALL_EXPERTS_FUNCTIONS.register("yamoe", yamoe.experts_forward)
    model = AutoModelForCausalLM.from_pretrained(..., experts_implementation="yamoe")
"""

import torch
import torch.nn.functional as F

from ._ops import add_op_namespace_prefix, ops

IS_ROCM = torch.version.hip is not None


def _defaults():
    """Per-backend dispatch and launch defaults, tuned on MI300X (ROCm), B200 (sm100) and RTX 4090 (sm89).

    The C++ op picks its path from the average rows per expert (S / E, S = T * K pairs):
      GEMV decode kernel (matrix-core tile for blocks with >= mfma_min rows) below tile_min_rows,
      16-row matrix-core tiles up to tile16_max_rows, 64-row tiles above.
    Past grouped_mm_min_rows a dense grouped GEMM (hipBLASLt / CUTLASS) is faster, so we hand off.
    """
    if IS_ROCM:
        return dict(cfg=(1024, 64, 16, 1, 1024, 64, 16, 1), mfma_min=5, tile_min_rows=1.0, tile16_max_rows=24.0,
                    grouped_mm_min_rows=384.0, small_max_pairs=None)
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    cfg = (512, 32, 16, 1, 512, 128, 8, 2)
    if major >= 9:  # Hopper / Blackwell: mma.sync tiles, CUTLASS grouped_mm is fast from ~16 rows per expert
        return dict(cfg=cfg, mfma_min=1 << 20, tile_min_rows=2.0, tile16_max_rows=float(1 << 30),
                    grouped_mm_min_rows=16.0, small_max_pairs=None)
    if major == 8:  # Ampere / Ada: grouped_mm has no fast path, our tiles win until long prefill
        return dict(cfg=cfg, mfma_min=1 << 20, tile_min_rows=0.5, tile16_max_rows=128.0,
                    grouped_mm_min_rows=384.0, small_max_pairs=None)
    return dict(cfg=(512, 32, 16, 1, 512, 128, 8, 2), mfma_min=1 << 20, tile_min_rows=float(1 << 30),
                tile16_max_rows=float(1 << 30), grouped_mm_min_rows=None, small_max_pairs=256)


DEFAULTS = _defaults()

# Gate activations the kernel implements: config name -> kernel id
ACTIVATIONS = {"silu": 0, "swish": 0, "gelu_pytorch_tanh": 1, "gelu_tanh": 1}


def _act_fn(act: int, x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh") if act == 1 else F.silu(x)


@torch.library.register_fake(add_op_namespace_prefix("fused_experts"))
def _fused_experts_fake(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, *args):
    return torch.empty_like(hidden_states)


def _grouped_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    if a.dtype == torch.bfloat16 and hasattr(F, "grouped_mm"):
        return F.grouped_mm(a, b, offs=offs)
    out = torch.empty(a.shape[0], b.shape[2], device=a.device, dtype=a.dtype)
    start = 0
    for e, end in enumerate(offs.tolist()):
        if end > start:
            torch.mm(a[start:end], b[e], out=out[start:end])
        start = end
    return out


def _grouped_mm_experts(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act: int = 0):
    """Dense grouped-GEMM path for long prefill, same math and ordering as transformers' grouped_mm forward."""
    T, H = hidden_states.shape
    K = top_k_index.shape[1]
    E = gate_up_proj.shape[0]
    ids, perm = torch.sort(top_k_index.reshape(-1))
    xs = hidden_states[perm // K]
    offs = torch.cumsum(torch.histc(ids.int(), bins=E, min=0, max=E - 1), 0, dtype=torch.int32)
    h = _grouped_mm(xs, gate_up_proj.transpose(1, 2), offs)
    gate, up = h.chunk(2, dim=-1)
    h = _grouped_mm(_act_fn(act, gate) * up, down_proj.transpose(1, 2), offs)
    h = h * top_k_weights.reshape(-1)[perm].unsqueeze(-1).to(h.dtype)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    return h[inv_perm].view(T, K, H).sum(dim=1).to(hidden_states.dtype)


def _use_kernel(num_pairs: int, num_experts: int) -> bool:
    d = DEFAULTS
    if d["small_max_pairs"] is not None:
        return num_pairs <= d["small_max_pairs"]
    return num_pairs <= d["grouped_mm_min_rows"] * num_experts


def _kernel(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act: int = 0):
    d = DEFAULTS
    return ops.fused_experts(
        hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype), gate_up_proj, down_proj, *d["cfg"],
        d["mfma_min"], d["tile_min_rows"], d["tile16_max_rows"], act,
    )


def fused_experts(
    hidden_states: torch.Tensor,  # [T, H]
    top_k_index: torch.Tensor,  # [T, K]
    top_k_weights: torch.Tensor,  # [T, K]
    gate_up_proj: torch.Tensor,  # [E, 2I, H], rows [0, I) gate, [I, 2I) up
    down_proj: torch.Tensor,  # [E, H, I]
    activation: str = "silu",
) -> torch.Tensor:
    """Gated experts: sum_k w[t, k] * down[e](act(gate[e] x) * up[e] x), no bias. act: silu or gelu_pytorch_tanh."""
    act = ACTIVATIONS[activation]
    if _use_kernel(top_k_index.numel(), gate_up_proj.shape[0]):
        return _kernel(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act)
    return _grouped_mm_experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype), gate_up_proj, down_proj, act)


_ACT_CLASSES = {"SiLUActivation": "silu", "SiLU": "silu", "GELUTanh": "gelu_pytorch_tanh", "PytorchGELUTanh": "gelu_pytorch_tanh"}


def _activation(module):
    """The gate activation the experts module actually applies (its act_fn), falling back to the config name."""
    name = _ACT_CLASSES.get(type(getattr(module, "act_fn", None)).__name__)
    if name is not None:
        return name
    cfg = getattr(module, "config", None)
    return getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None)


def _supported(module) -> bool:
    return (
        getattr(module, "has_gate", False)
        and not getattr(module, "has_bias", True)
        and not getattr(module, "is_transposed", True)
        and getattr(module, "is_concatenated", False)
        and not getattr(module, "has_post_expert_norm", False)
        and not getattr(module, "_is_expert_parallel", False)
        and _activation(module) in ACTIVATIONS
        and type(module)._apply_gate.__name__ == "_default_apply_gate"
        and module.gate_up_proj.dtype in (torch.bfloat16, torch.float16)
    )


def experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """transformers `ExpertsInterface` forward.

    Uses the yamoe kernel where it is faster, and transformers' own grouped_mm forward for long prefill and for
    layouts the kernel does not support, so those cases behave exactly like `experts_implementation="grouped_mm"`.
    """
    if _supported(module) and _use_kernel(top_k_index.numel(), module.gate_up_proj.shape[0]):
        return _kernel(
            hidden_states, top_k_index, top_k_weights, module.gate_up_proj, module.down_proj, ACTIVATIONS[_activation(module)]
        )
    from transformers.integrations.moe import grouped_mm_experts_forward

    return grouped_mm_experts_forward(module, hidden_states, top_k_index, top_k_weights)
