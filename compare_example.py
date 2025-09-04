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
import numpy as np
import sys

# Set seeds and deterministic flags for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.set_printoptions(precision=4)

load_method = 2  # 1: sym, 2: local, 3: hf

if load_method == 1:
    sys.path.insert(0, "./torch-ext")
    import yamoe
elif load_method == 2:
    yamoe = get_local_kernel(Path("result"), "yamoe")
elif load_method == 3:
    yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")

binned_experts_ref = yamoe.vendored.yamoe_ref.binned_experts_ref
GptOssExperts = yamoe.vendored.gpt_oss_mlp.GptOssExperts

# Configuration
batch_size, seq_len, hidden_dim = 4, 1024, 2880
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
# gate_up_proj = torch.randn(num_experts, hidden_dim, 2 * hidden_dim).cuda()
gate_up_proj_bias = torch.zeros(num_experts, 2 * hidden_dim).cuda()
# down_proj = torch.randn(num_experts, hidden_dim, hidden_dim).cuda()
down_proj_bias = torch.zeros(num_experts, hidden_dim).cuda()
# routing_weights = routing_weights.cuda()
router_indices = flat_indices.cuda()

gate_up_proj = torch.empty(num_experts, hidden_dim, 2 * hidden_dim, device="cuda")
down_proj = torch.empty(num_experts, hidden_dim, hidden_dim, device="cuda")
torch.nn.init.trunc_normal_(gate_up_proj, std=0.02)
torch.nn.init.trunc_normal_(down_proj, std=0.02)

routing_weights = routing_weights.to(dtype=torch.float32, device="cuda")
expert_capacity = batch_seq * top_k // num_experts * 2


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
        expert_capacity,
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
        expert_capacity,
        num_experts,
        top_k,
    )

torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1e3
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

# Store kernel results
kernel_output = output.clone()
kernel_time = elapsed_ms
kernel_memory = peak_mem_mb

## OPTIONAL
# Compare to reference implementation
config = type("Config", (), {})()
config.hidden_size = hidden_dim
config.intermediate_size = 4 * hidden_dim
config.num_local_experts = num_experts

model = GptOssExperts(config)

# set the weights and biases from above to the reference model
model.gate_up_proj.data = gate_up_proj
model.gate_up_proj_bias.data = gate_up_proj_bias
model.down_proj.data = down_proj
model.down_proj_bias.data = down_proj_bias

model = model.cuda()
model.eval()

# Warmup
for _ in range(5):
    _ = model(hidden_states, router_indices, routing_weights)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.no_grad():
    ref_output = model(hidden_states, router_indices, routing_weights)

torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1e3
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

# Store reference results
ref_time = elapsed_ms
ref_memory = peak_mem_mb

# Reshape reference output to match kernel output
ref_output_reshaped = ref_output.view(kernel_output.shape)

# Test yamoe_ref implementation
expert_capacity = batch_seq * top_k // num_experts * 2  # Generous capacity

# Warmup
for _ in range(5):
    _ = binned_experts_ref(
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        expert_capacity,
    )

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.no_grad():
    yamoe_ref_output = binned_experts_ref(
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        expert_capacity,
    )

torch.cuda.synchronize()
yamoe_ref_time = (time.perf_counter() - start) * 1e3
yamoe_ref_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

# Reshape yamoe_ref output to match kernel output
yamoe_ref_output_reshaped = yamoe_ref_output.view(kernel_output.shape)

# Calculate similarity metrics between kernel and reference
mse_kernel_ref = torch.nn.functional.mse_loss(kernel_output, ref_output_reshaped).item()
mae_kernel_ref = torch.nn.functional.l1_loss(kernel_output, ref_output_reshaped).item()

# Cosine similarity
kernel_flat = kernel_output.view(-1)
ref_flat = ref_output_reshaped.view(-1)
yamoe_ref_flat = yamoe_ref_output_reshaped.view(-1)
cosine_sim_kernel_ref = torch.nn.functional.cosine_similarity(
    kernel_flat.unsqueeze(0), ref_flat.unsqueeze(0)
).item()

# Relative error (L2 norm of difference / L2 norm of reference)
diff_norm_kernel_ref = torch.norm(kernel_output - ref_output_reshaped).item()
ref_norm = torch.norm(ref_output_reshaped).item()
rel_error_kernel_ref = diff_norm_kernel_ref / ref_norm if ref_norm > 0 else float("inf")

# Max absolute difference
max_abs_diff_kernel_ref = torch.max(
    torch.abs(kernel_output - ref_output_reshaped)
).item()

# Calculate similarity metrics between kernel and yamoe_ref
mse_kernel_yamoe = torch.nn.functional.mse_loss(
    kernel_output, yamoe_ref_output_reshaped
).item()
mae_kernel_yamoe = torch.nn.functional.l1_loss(
    kernel_output, yamoe_ref_output_reshaped
).item()
cosine_sim_kernel_yamoe = torch.nn.functional.cosine_similarity(
    kernel_flat.unsqueeze(0), yamoe_ref_flat.unsqueeze(0)
).item()
diff_norm_kernel_yamoe = torch.norm(kernel_output - yamoe_ref_output_reshaped).item()
yamoe_ref_norm = torch.norm(yamoe_ref_output_reshaped).item()
rel_error_kernel_yamoe = (
    diff_norm_kernel_yamoe / yamoe_ref_norm if yamoe_ref_norm > 0 else float("inf")
)
max_abs_diff_kernel_yamoe = torch.max(
    torch.abs(kernel_output - yamoe_ref_output_reshaped)
).item()

# Calculate similarity metrics between reference and yamoe_ref
mse_ref_yamoe = torch.nn.functional.mse_loss(
    ref_output_reshaped, yamoe_ref_output_reshaped
).item()
mae_ref_yamoe = torch.nn.functional.l1_loss(
    ref_output_reshaped, yamoe_ref_output_reshaped
).item()
cosine_sim_ref_yamoe = torch.nn.functional.cosine_similarity(
    ref_flat.unsqueeze(0), yamoe_ref_flat.unsqueeze(0)
).item()
diff_norm_ref_yamoe = torch.norm(ref_output_reshaped - yamoe_ref_output_reshaped).item()
rel_error_ref_yamoe = (
    diff_norm_ref_yamoe / yamoe_ref_norm if yamoe_ref_norm > 0 else float("inf")
)
max_abs_diff_ref_yamoe = torch.max(
    torch.abs(ref_output_reshaped - yamoe_ref_output_reshaped)
).item()

# Print comparison table
print("\n" + "=" * 110)
print(
    f"{'METRIC':<20} {'KERNEL':<15} {'REFERENCE':<15} {'YAMOE_REF':<15} {'KERNEL SPEEDUP':<20} {'REF SPEEDUP':<15}"
)
print("=" * 110)

print(
    f"{'Sum':<20} {kernel_output.sum().item():<15.4f} {ref_output_reshaped.sum().item():<15.4f} {yamoe_ref_output_reshaped.sum().item():<15.4f} {'N/A':<20} {'N/A':<15}"
)
print(
    f"{'Min':<20} {kernel_output.min().item():<15.4f} {ref_output_reshaped.min().item():<15.4f} {yamoe_ref_output_reshaped.min().item():<15.4f} {'N/A':<20} {'N/A':<15}"
)
print(
    f"{'Max':<20} {kernel_output.max().item():<15.4f} {ref_output_reshaped.max().item():<15.4f} {yamoe_ref_output_reshaped.max().item():<15.4f} {'N/A':<20} {'N/A':<15}"
)
print(
    f"{'Norm (L2)':<20} {kernel_output.norm().item():<15.4f} {ref_output_reshaped.norm().item():<15.4f} {yamoe_ref_output_reshaped.norm().item():<15.4f} {'N/A':<20} {'N/A':<15}"
)
print(
    f"{'Std':<20} {kernel_output.std().item():<15.4f} {ref_output_reshaped.std().item():<15.4f} {yamoe_ref_output_reshaped.std().item():<15.4f} {'N/A':<20} {'N/A':<15}"
)

print("-" * 110)
print(
    f"{'Time (ms)':<20} {kernel_time:<15.3f} {ref_time:<15.3f} {yamoe_ref_time:<15.3f} {yamoe_ref_time / kernel_time:<20.2f}x {yamoe_ref_time / ref_time:<15.2f}x"
)
print(
    f"{'Memory (MB)':<20} {kernel_memory:<15.2f} {ref_memory:<15.2f} {yamoe_ref_memory:<15.2f} {yamoe_ref_memory / kernel_memory:<20.2f}x {yamoe_ref_memory / ref_memory:<15.2f}x"
)

print("-" * 110)
print("SIMILARITY METRICS (vs KERNEL)")
print("-" * 110)
print(
    f"{'METRIC':<20} {'KERNEL vs REF':<20} {'KERNEL vs YAMOE_REF':<20} {'REF vs YAMOE_REF':<20}"
)
print("-" * 110)
print(
    f"{'MSE':<20} {mse_kernel_ref:<20.6e} {mse_kernel_yamoe:<20.6e} {mse_ref_yamoe:<20.6e}"
)
print(
    f"{'MAE':<20} {mae_kernel_ref:<20.6e} {mae_kernel_yamoe:<20.6e} {mae_ref_yamoe:<20.6e}"
)
print(
    f"{'Cosine Similarity':<20} {cosine_sim_kernel_ref:<20.6f} {cosine_sim_kernel_yamoe:<20.6f} {cosine_sim_ref_yamoe:<20.6f}"
)
print(
    f"{'Relative Error':<20} {rel_error_kernel_ref:<20.6e} {rel_error_kernel_yamoe:<20.6e} {rel_error_ref_yamoe:<20.6e}"
)
print(
    f"{'Max Abs Diff':<20} {max_abs_diff_kernel_ref:<20.6e} {max_abs_diff_kernel_yamoe:<20.6e} {max_abs_diff_ref_yamoe:<20.6e}"
)

print("-" * 110)
print("FIRST 10 ELEMENTS COMPARISON")
print("-" * 110)


# Get first N elements as numpy arrays for nice display
N = 10
kernel_first_10 = kernel_flat[:N].cpu().numpy()
ref_first_10 = ref_flat[:N].cpu().numpy()
yamoe_ref_first_10 = yamoe_ref_flat[:N].cpu().numpy()
diff_kernel_ref = kernel_first_10 - ref_first_10
diff_kernel_yamoe = kernel_first_10 - yamoe_ref_first_10

print(
    f"{'INDEX':<5} {'KERNEL':<12} {'REFERENCE':<12} {'YAMOE_REF':<12} {'K-R DIFF':<12} {'K-Y DIFF':<12}"
)
print("-" * 70)
for i in range(N):
    print(
        f"{i:<5} {kernel_first_10[i]:<12.6f} {ref_first_10[i]:<12.6f} {yamoe_ref_first_10[i]:<12.6f} {diff_kernel_ref[i]:<12.6f} {diff_kernel_yamoe[i]:<12.6f}"
    )

print("=" * 110)
