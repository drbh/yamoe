"""Experts forward for the transformers experts layout, on CUDA and ROCm.

`fused_experts` is the functional entry point. `experts_forward` matches the transformers
`ExpertsInterface` signature, so it can be registered as an experts implementation:

    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    ALL_EXPERTS_FUNCTIONS.register("yamoe", yamoe.experts_forward)
    model = AutoModelForCausalLM.from_pretrained(..., experts_implementation="yamoe")
"""

import torch
import torch.nn.functional as F

from ._ops import add_op_namespace_prefix, ops

IS_ROCM = torch.version.hip is not None

# The C++ op picks its own path from the average rows per expert (S / E): a GEMV / MFMA decode kernel, then
# 16-row and 64-row matrix-core tiles (ROCm). Past these limits a dense grouped GEMM is faster.
#   ROCm: tiles are used up to GROUPED_MM_MIN_ROWS rows per expert, then hipBLASLt grouped_mm (long prefill).
#   CUDA: no tile path yet, so the decode kernel is used up to CUDA_SMALL_MAX_PAIRS pairs, then grouped_mm.
GROUPED_MM_MIN_ROWS = 384
CUDA_SMALL_MAX_PAIRS = 256  # B200: decode kernel wins up to ~32 tokens at top-8, CUTLASS grouped_mm above


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


def _grouped_mm_experts(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj):
    """Dense grouped-GEMM path for long prefill (and the CUDA fallback)."""
    T, H = hidden_states.shape
    K = top_k_index.shape[1]
    E = gate_up_proj.shape[0]
    ids, perm = torch.sort(top_k_index.reshape(-1))
    xs = hidden_states[perm // K]
    offs = torch.cumsum(torch.histc(ids.int(), bins=E, min=0, max=E - 1), 0, dtype=torch.int32)
    h = _grouped_mm(xs, gate_up_proj.transpose(1, 2), offs)
    gate, up = h.chunk(2, dim=-1)
    h = _grouped_mm(F.silu(gate) * up, down_proj.transpose(1, 2), offs)
    h = h * top_k_weights.reshape(-1)[perm].unsqueeze(-1).to(h.dtype)
    y = torch.empty_like(h)
    y[perm] = h
    return y.view(T, K, H).sum(dim=1, dtype=torch.float32).to(hidden_states.dtype)


def fused_experts(
    hidden_states: torch.Tensor,  # [T, H]
    top_k_index: torch.Tensor,  # [T, K]
    top_k_weights: torch.Tensor,  # [T, K]
    gate_up_proj: torch.Tensor,  # [E, 2I, H], rows [0, I) gate, [I, 2I) up
    down_proj: torch.Tensor,  # [E, H, I]
) -> torch.Tensor:
    """SiLU-gated experts: sum_k w[t, k] * down[e](silu(gate[e] x) * up[e] x), no bias."""
    weights = top_k_weights.to(hidden_states.dtype)
    S = top_k_index.numel()
    E = gate_up_proj.shape[0]
    if IS_ROCM:
        use_kernel = S <= GROUPED_MM_MIN_ROWS * E
    else:
        use_kernel = S <= CUDA_SMALL_MAX_PAIRS
    if use_kernel:
        return ops.fused_experts(hidden_states, top_k_index, weights, gate_up_proj, down_proj)
    return _grouped_mm_experts(hidden_states, top_k_index, weights, gate_up_proj, down_proj)


def _supported(module) -> bool:
    return (
        getattr(module, "has_gate", False)
        and not getattr(module, "has_bias", True)
        and not getattr(module, "is_transposed", True)
        and getattr(module, "is_concatenated", False)
        and not getattr(module, "has_post_expert_norm", False)
        and not getattr(module, "_is_expert_parallel", False)
        and getattr(getattr(module, "config", None), "hidden_act", None) == "silu"
        and type(module)._apply_gate.__name__ == "_default_apply_gate"
        and module.gate_up_proj.dtype in (torch.bfloat16, torch.float16)
    )


def experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """transformers `ExpertsInterface` forward; falls back to grouped_mm for unsupported layouts."""
    if not _supported(module):
        from transformers.integrations.moe import grouped_mm_experts_forward

        return grouped_mm_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    return fused_experts(hidden_states, top_k_index, top_k_weights, module.gate_up_proj, module.down_proj)
