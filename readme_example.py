# /// script
# requires-python = "==3.10"
# dependencies = ["torch==2.7.0", "triton", "numpy", "kernels"]
# [tool.uv.sources]
# kernels = { git = "https://github.com/huggingface/kernels.git" }
# ///

import time
import torch
from kernels import get_kernel, get_local_kernel
from pathlib import Path
from torch.nn import functional as F

# Set seeds and deterministic flags for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")

# Configuration
batch_size, seq_len, hidden_dim = 16, 256, 2880
num_experts, top_k = 8, 2

# Create routing weights
logits = torch.randn(batch_size, seq_len, num_experts)
probs = F.softmax(logits, dim=-1)
weights, indices = torch.topk(probs, top_k, dim=-1)

batch_seq = batch_size * seq_len
routing_weights = torch.zeros(batch_seq, num_experts, dtype=weights.dtype)
flat_indices, flat_weights = indices.reshape(-1, top_k), weights.reshape(-1, top_k)
batch_indices = torch.arange(batch_seq).unsqueeze(1).expand(-1, top_k)
routing_weights[batch_indices, flat_indices] = flat_weights

# Create model tensors
hidden_states = torch.randn(batch_size, seq_len, hidden_dim).cuda()
gate_up_proj = torch.randn(num_experts, hidden_dim, 2 * hidden_dim).cuda()
gate_up_proj_bias = torch.zeros(num_experts, 2 * hidden_dim).cuda()
down_proj = torch.randn(num_experts, hidden_dim, hidden_dim).cuda()
down_proj_bias = torch.zeros(num_experts, hidden_dim).cuda()
routing_weights = routing_weights.cuda()
router_indices = flat_indices.cuda()

# Warmup
for _ in range(5):
    _ = yamoe.experts(
        hidden_states.view(-1, hidden_dim),
        router_indices,
        routing_weights.view(-1, num_experts),
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        seq_len,
        num_experts,
        top_k,
    )

# Benchmark
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.no_grad():
    output = yamoe.experts(
        hidden_states.view(-1, hidden_dim),
        router_indices,
        routing_weights.view(-1, num_experts),
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        seq_len,
        num_experts,
        top_k,
    )

torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1e3
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

print(
    f"Output: sum={output.sum().item():.1f}, min={output.min().item():.1f}, max={output.max().item():.1f}"
)
print(f"First 3: {output.view(-1)[:3].tolist()}")
print(f"Time: {elapsed_ms:.1f}ms, Memory: {peak_mem_mb:.0f}MB")
