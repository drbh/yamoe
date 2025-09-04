# /// script
# requires-python = "==3.10"
# dependencies = ["torch==2.7.0", "triton", "numpy", "kernels"]
# [tool.uv.sources]
# kernels = { git = "https://github.com/huggingface/kernels.git" }
# ///

import torch
import sys
import time
from kernels import get_kernel, get_local_kernel
from pathlib import Path

load_method = 3 # 1: sym, 2: local, 3: hf

if load_method == 1:
    sys.path.insert(0, "./torch-ext")
    import yamoe
elif load_method == 2:
    yamoe = get_local_kernel(Path("result"), "yamoe")
elif load_method == 3:
    yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")

torch.manual_seed(42)


def benchmark_forward(model, x, tag: str, iters: int = 10, warmup: int = 10):
    x_local = x.detach().clone().requires_grad_(False)

    for _ in range(warmup):
        out = model(x_local)
        out = out[0] if isinstance(out, tuple) else out

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = model(x_local)
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) * 1e3 / iters
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB

    print(f"[{tag}] fwd: {fwd_ms:.2f} ms | peak mem: {peak_mem:.2f} GB")
    return fwd_ms


def main():
    ref_moe_cls = yamoe.vendored.gpt_oss_mlp.GptOssMLP
    new_moe_cls = yamoe.Yamoe

    batch_size, seq_len, hidden_dim = 1, 1024, 2880
    num_experts, top_k = 32, 4

    print("\nInput parameters:")
    print(f" Batch size: {batch_size}")
    print(f" Seq len: {seq_len}")
    print(f" Hidden dim: {hidden_dim}")
    print(f" Num experts: {num_experts}")
    print(f" Top-k: {top_k}")

    config = type("Config", (), {})()
    config.hidden_size = hidden_dim
    config.intermediate_size = hidden_dim
    config.num_local_experts = num_experts
    config.num_experts_per_tok = top_k
    ref_moe = ref_moe_cls(config)

    print("\nModel:")
    print(ref_moe)

    for p in ref_moe.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
    ref_moe = ref_moe.cuda()
    ref_moe = ref_moe.eval()

    # Test reference implementation
    print("\nReference Implementation")

    # Small warmup
    print(" Warming up...")
    for _ in range(3):
        _ = ref_moe(x)
    torch.cuda.synchronize()

    x_ref = x.detach().requires_grad_(False)
    ref_output = ref_moe(x_ref)
    out = ref_output[0] if isinstance(ref_output, tuple) else ref_output
    print(f" Input shape: {x_ref.shape}")
    print(f" Output shape: {out.shape}")
    print(
        f" Output mean: {out.mean():.6f}, std: {out.std():.6f}, norm: {out.norm():.6f}"
    )

    benchmark_forward(ref_moe, x, tag="reference", warmup=10, iters=20)

    # Switch to YAMOE forward
    print("\nYAMOE Implementation")
    ref_moe.forward = new_moe_cls.forward.__get__(ref_moe)
    ref_moe._routing_weights_buffer = None
    ref_moe._batch_indices_buffer = None
    ref_moe._last_batch_seq = None
    ref_moe._last_num_experts = None
    ref_moe.enable_router_grads = False
    ref_moe.num_experts = num_experts
    ref_moe.top_k = top_k

    # Small warmup
    print(" Warming up...")
    for _ in range(3):
        _ = ref_moe(x)
    torch.cuda.synchronize()

    x_cuda = x.detach().requires_grad_(False)
    cuda_output = ref_moe(x_cuda)
    out = cuda_output[0] if isinstance(cuda_output, tuple) else cuda_output
    print(f" Input shape: {x_cuda.shape}")
    print(f" Output shape: {out.shape}")
    print(
        f" Output mean: {out.mean():.6f}, std: {out.std():.6f}, norm: {out.norm():.6f}"
    )

    benchmark_forward(ref_moe, x, tag="yamoe", warmup=10, iters=20)


if __name__ == "__main__":
    main()
