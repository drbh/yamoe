"""Correctness of yamoe.fused_experts and every path of the fused_experts op against an fp32 reference.

Loads the local build when YAMOE_BUILD points at a directory containing `build/<variant>`, else the Hub kernel.
"""

import importlib
import os
from pathlib import Path

import pytest
import torch

import kernels

if os.environ.get("YAMOE_BUILD"):
    yamoe = kernels.get_local_kernel(Path(os.environ["YAMOE_BUILD"]), trust_remote_code=True)
else:
    yamoe = kernels.get_kernel("drbh/yamoe", version=2, trust_remote_code=True)
experts_mod = importlib.import_module(yamoe.__name__ + ".experts")

DEVICE = "cuda"
CFG = (1024, 64, 16, 1, 1024, 64, 16, 1)  # launch config defaults


def reference(x, idx, w, w_gu, w_dn, activation="silu"):
    """fp32 reference, vectorized per expert."""
    act = (lambda v: torch.nn.functional.gelu(v, approximate="tanh")) if activation == "gelu_pytorch_tanh" else torch.nn.functional.silu
    T, H = x.shape
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in idx.unique().tolist():
        t, j = (idx == e).nonzero(as_tuple=True)
        g, u = (xf[t] @ w_gu[e].float().T).chunk(2, dim=-1)
        y = (act(g) * u) @ w_dn[e].float().T
        out.index_add_(0, t, y * w[t, j].float().unsqueeze(-1))
    return out


def make(T, E, H, I, K, dtype, skew=0.0, seed=0):
    g = torch.Generator(DEVICE).manual_seed(seed)
    x = torch.randn(T, H, device=DEVICE, dtype=dtype, generator=g)
    w_gu = torch.randn(E, 2 * I, H, device=DEVICE, dtype=dtype, generator=g) * 0.02
    w_dn = torch.randn(E, H, I, device=DEVICE, dtype=dtype, generator=g) * 0.02
    logits = torch.randn(T, E, device=DEVICE, generator=g) + skew * torch.randn(E, device=DEVICE, generator=g)
    w, idx = torch.topk(logits.softmax(-1), K)
    return x, idx, (w / w.sum(-1, keepdim=True)).to(dtype), w_gu, w_dn


def check(out, ref, dtype):
    assert out.dtype == dtype and out.shape == ref.shape
    torch.testing.assert_close(out.float(), ref, atol=2e-2 * ref.abs().max().item(), rtol=2e-2)


SHAPES = [
    # E, H, I, K: Kimi-Linear, Qwen3-30B-A3B, Qwen3-Next, Qwen3-235B (H=4096 -> 4-row decode blocks), gpt-oss-ish, tiny
    (256, 2304, 1024, 8),
    (128, 2048, 768, 8),
    (512, 2048, 512, 10),
    (128, 4096, 1536, 8),
    (32, 2880, 2880, 4),
    (8, 64, 32, 2),
]


@pytest.mark.parametrize("E,H,I,K", SHAPES)
@pytest.mark.parametrize("T", [1, 3, 8, 16, 33, 200, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_fused_experts(E, H, I, K, T, dtype, skew):
    x, idx, w, w_gu, w_dn = make(T, E, H, I, K, dtype, skew)
    check(yamoe.fused_experts(x, idx, w, w_gu, w_dn), reference(x, idx, w, w_gu, w_dn), dtype)


# Force each path of the C++ op: (mfma_min, tile_min_rows, tile16_max_rows). Tiles and MFMA are ROCm only;
# on CUDA these all run the GEMV decode kernel, which is still checked.
PATHS = {
    "decode_gemv": (99, 1 << 20, 24),
    "decode_mfma": (1, 1 << 20, 24),
    "tile16": (5, 0, 1 << 20),
    "tile64": (5, 0, 0),
}


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("E,H,I,K", [(128, 2048, 768, 8), (64, 512, 256, 4), (16, 96, 64, 2)])
@pytest.mark.parametrize("T", [1, 5, 64, 300])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kernel_paths(path, E, H, I, K, T, dtype):
    mfma_min, tile_min_rows, tile16_max_rows = PATHS[path]
    x, idx, w, w_gu, w_dn = make(T, E, H, I, K, dtype, skew=2.0)
    out = yamoe.ops.fused_experts(x, idx, w, w_gu, w_dn, *CFG, mfma_min, tile_min_rows, tile16_max_rows)
    check(out, reference(x, idx, w, w_gu, w_dn), dtype)


# Gemma-4 26B-A4B shape (128 experts, H=2816, I=704, top-8) with its GELU-tanh gate, through every path
@pytest.mark.parametrize("T", [1, 5, 40, 300, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gelu_tanh(T, dtype):
    x, idx, w, w_gu, w_dn = make(T, 128, 2816, 704, 8, dtype, skew=2.0)
    ref = reference(x, idx, w, w_gu, w_dn, "gelu_pytorch_tanh")
    check(yamoe.fused_experts(x, idx, w, w_gu, w_dn, activation="gelu_pytorch_tanh"), ref, dtype)
    for mfma_min, tile_min_rows, tile16_max_rows in PATHS.values():
        out = yamoe.ops.fused_experts(x, idx, w, w_gu, w_dn, *CFG, mfma_min, tile_min_rows, tile16_max_rows, 1)
        check(out, ref, dtype)
    check(experts_mod._grouped_mm_experts(x, idx, w, w_gu, w_dn, 1), ref, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_mm_fallback(dtype):
    x, idx, w, w_gu, w_dn = make(700, 32, 256, 128, 4, dtype)
    check(experts_mod._grouped_mm_experts(x, idx, w, w_gu, w_dn), reference(x, idx, w, w_gu, w_dn), dtype)


def test_int32_routing_indices():
    x, idx, w, w_gu, w_dn = make(8, 64, 512, 256, 4, torch.bfloat16)
    check(yamoe.ops.fused_experts(x, idx.int(), w, w_gu, w_dn), reference(x, idx, w, w_gu, w_dn), torch.bfloat16)


@pytest.mark.parametrize("T", [4, 64])
def test_cuda_graph_capture(T):
    x, idx, w, w_gu, w_dn = make(T, 64, 512, 256, 4, torch.bfloat16)
    expected = yamoe.fused_experts(x, idx, w, w_gu, w_dn)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        yamoe.fused_experts(x, idx, w, w_gu, w_dn)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = yamoe.fused_experts(x, idx, w, w_gu, w_dn)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)


def test_torch_compile():
    x, idx, w, w_gu, w_dn = make(4, 64, 512, 256, 4, torch.bfloat16)
    expected = yamoe.fused_experts(x, idx, w, w_gu, w_dn)
    compiled = torch.compile(yamoe.fused_experts, fullgraph=True)
    torch.testing.assert_close(compiled(x, idx, w, w_gu, w_dn), expected)
