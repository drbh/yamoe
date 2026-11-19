"""Experts forward for the transformers experts layout, on CUDA and ROCm.

`fused_experts` is the functional entry point (gate activation SiLU or GELU tanh, bf16 / f16 or FP8 block-scaled
weights). `experts_forward` and `fp8_experts_forward` match the transformers experts interfaces:

    yamoe.register()  # registers "yamoe" for bf16 / f16 experts and fine-grained FP8 experts
    model = AutoModelForCausalLM.from_pretrained(..., experts_implementation="yamoe")
"""

from __future__ import annotations

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


def _kernel(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act: int = 0, scales=None):
    d = DEFAULTS
    gs, ds, bn, bk = scales if scales is not None else (None, None, 128, 128)
    return ops.fused_experts(
        hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype), gate_up_proj, down_proj, *d["cfg"],
        d["mfma_min"], d["tile_min_rows"], d["tile16_max_rows"], act, gs, ds, bn, bk,
    )


def fused_experts(
    hidden_states: torch.Tensor,  # [T, H]
    top_k_index: torch.Tensor,  # [T, K]
    top_k_weights: torch.Tensor,  # [T, K]
    gate_up_proj: torch.Tensor,  # [E, 2I, H], rows [0, I) gate, [I, 2I) up
    down_proj: torch.Tensor,  # [E, H, I]
    activation: str = "silu",
    gate_up_scale: torch.Tensor | None = None,  # FP8 weights: [E, ceil(2I / bn), ceil(H / bk)] fp32
    down_scale: torch.Tensor | None = None,  # FP8 weights: [E, ceil(H / bn), ceil(I / bk)] fp32
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Gated experts: sum_k w[t, k] * down[e](act(gate[e] x) * up[e] x), no bias. act: silu or gelu_pytorch_tanh.

    Weights are the activation dtype, or float8_e4m3fn with fp32 block scales (dequantized in the kernel).
    """
    act = ACTIVATIONS[activation]
    if gate_up_proj.dtype == torch.float8_e4m3fn:
        return _kernel(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act,
                       (gate_up_scale, down_scale, block_size[0], block_size[1]))
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


# ---------------------------------------------------------------------------------------------------------------------
# Fine-grained FP8 experts (transformers `FP8Experts`, e.g. Qwen3.6-35B-A3B-FP8): weight-only dequant in the kernel.
# ROCm: the kernel wins at every size (MI300X: 2-7x over the Triton FP8 paths). Hopper / Blackwell: native FP8 tensor
# cores in finegrained-fp8 win past small decode batches (also on Ada), so we hand off to transformers' FP8 forwards.
# ---------------------------------------------------------------------------------------------------------------------
def _fp8_kernel_max_pairs():
    # ROCm: always (MI300X Qwen3-30B-A3B-FP8: 1.7-4.7x decode, 1.2-1.7x prefill over the transformers FP8 paths).
    # CUDA: single-token decode only; on B200 / RTX 4090 finegrained-fp8's FP8 tensor-core paths are as fast or faster
    # from a few tokens up.
    return None if IS_ROCM else 8


FP8_KERNEL_MAX_PAIRS = _fp8_kernel_max_pairs()


def _fp8_supported(module, hidden_states) -> bool:
    bs = getattr(module, "block_size", None)
    if not (
        getattr(module, "has_gate", False)
        and not getattr(module, "has_bias", True)
        and not getattr(module, "is_transposed", False)
        and getattr(module, "is_concatenated", True)
        and not getattr(module, "_is_expert_parallel", False)
        and getattr(module, "activation_scheme", None) == "dynamic"
        and getattr(module, "swiglu_alpha", None) is None
        and getattr(module, "limit", None) is None
        and bs is not None
        and len(bs) == 2
        and bs[1] % 16 == 0
        and _activation(module) in ACTIVATIONS
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and getattr(module, "gate_up_proj", None) is not None
        and module.gate_up_proj.dtype == torch.float8_e4m3fn
        and module.gate_up_proj_scale_inv.dtype == torch.float32
        and module.down_proj_scale_inv.dtype == torch.float32
    ):
        return False
    E, N2, H = module.gate_up_proj.shape
    I = module.down_proj.shape[2]
    cdiv = lambda a, b: (a + b - 1) // b  # noqa: E731
    return (
        H % 16 == 0
        and I % 16 == 0
        and tuple(module.gate_up_proj_scale_inv.shape) == (E, cdiv(N2, bs[0]), cdiv(H, bs[1]))
        and tuple(module.down_proj_scale_inv.shape) == (E, cdiv(H, bs[0]), cdiv(I, bs[1]))
    )


def fp8_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """transformers `FP8ExpertsInterface` forward for fine-grained FP8 experts."""
    S = top_k_index.numel()
    if _fp8_supported(module, hidden_states) and (FP8_KERNEL_MAX_PAIRS is None or S <= FP8_KERNEL_MAX_PAIRS):
        bs = module.block_size
        return _kernel(
            hidden_states, top_k_index, top_k_weights, module.gate_up_proj, module.down_proj,
            ACTIVATIONS[_activation(module)],
            (module.gate_up_proj_scale_inv, module.down_proj_scale_inv, int(bs[0]), int(bs[1])),
        )
    from transformers.integrations.finegrained_fp8 import fp8_batched_mm_experts_forward, fp8_grouped_mm_experts_forward

    if S <= 128 and getattr(module, "activation_scheme", "dynamic") == "dynamic":
        return fp8_batched_mm_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    return fp8_grouped_mm_experts_forward(module, hidden_states, top_k_index, top_k_weights)


def register(name: str = "yamoe") -> None:
    """Register yamoe as experts implementation `name` for bf16 / f16 and fine-grained FP8 experts in transformers."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS.register(name, experts_forward)
    try:
        from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS
    except ImportError:
        return
    ALL_FP8_EXPERTS_FUNCTIONS.register(name, fp8_experts_forward)
