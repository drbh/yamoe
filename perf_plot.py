# /// script
# requires-python = "==3.10"
# dependencies = ["torch==2.7.0", "triton", "numpy", "kernels", "matplotlib"]
# [tool.uv.sources]
# kernels = { git = "https://github.com/huggingface/kernels.git" }
# ///

import time
import torch
from kernels import get_local_kernel, get_kernel
from pathlib import Path
from torch.nn import functional as F
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# sys.path.insert(0, "./torch-ext")
# import yamoe
# import yamoe.reference as reference

yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")
reference = yamoe.reference

# Setup
torch.manual_seed(0)

# Parameter combinations to test
configs = [
    {"seq_len": 512, "hidden_dim": 2880, "num_experts": 32, "top_k": 4},
    {"seq_len": 1024, "hidden_dim": 2880, "num_experts": 32, "top_k": 4},
    {"seq_len": 512, "hidden_dim": 1024, "num_experts": 32, "top_k": 4},
    {"seq_len": 512, "hidden_dim": 2880, "num_experts": 16, "top_k": 2},
    {"seq_len": 2048, "hidden_dim": 1024, "num_experts": 16, "top_k": 2},
    {"seq_len": 768, "hidden_dim": 2048, "num_experts": 64, "top_k": 8},
]

# Strategic batch sizes: small (1,2), medium (4,8), large (16,32), extra large (64)
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
all_results = []

# Test each configuration
for config_idx, config in enumerate(configs):
    seq_len = config["seq_len"]
    hidden_dim = config["hidden_dim"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]

    print(f"\n{'=' * 70}")
    print(
        f"Config {config_idx + 1}: seq={seq_len}, hidden={hidden_dim}, experts={num_experts}, top_k={top_k}"
    )
    print(f"{'=' * 70}")

    yamoe_times = []
    reference_times = []
    yamoe_memory = []
    reference_memory = []
    speedups = []

    # Iterate over batch sizes
    for batch_size in batch_sizes:
        print(f"\nBatch size = {batch_size}")

        try:
            # Create logits for this batch size
            logits = torch.randn(batch_size, seq_len, num_experts)

            # Inline routing creation
            weights, indices = torch.topk(logits, top_k, dim=-1)
            weights = F.softmax(weights, dim=-1)
            batch_seq = batch_size * seq_len
            routing_weights = torch.zeros(
                batch_seq, num_experts, device=logits.device, dtype=weights.dtype
            )
            flat_indices, flat_weights = (
                indices.reshape(-1, top_k),
                weights.reshape(-1, top_k),
            )
            batch_indices = (
                torch.arange(batch_seq, device=logits.device)
                .unsqueeze(1)
                .expand(-1, top_k)
            )
            routing_weights[batch_indices, flat_indices] = flat_weights
            router_indices = flat_indices

            # Create tensors and convert to CUDA half precision
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim).cuda().half()
            gate_up_proj = (
                torch.randn(num_experts, hidden_dim, 2 * hidden_dim).cuda().half()
            )
            gate_up_proj_bias = torch.ones(num_experts, 2 * hidden_dim).cuda().half()
            down_proj = torch.randn(num_experts, hidden_dim, hidden_dim).cuda().half()
            down_proj_bias = torch.ones(num_experts, hidden_dim).cuda().half()
            logits, routing_weights = (
                logits.cuda().half(),
                routing_weights.cuda().half(),
            )
            router_indices = router_indices.cuda()

            # Test Yamoe kernel first
            yamoe_success = True
            yamoe_time = None
            yamoe_mem = None

            try:
                # Warmup runs for yamoe
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

                # Time and measure memory for yamoe kernel
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                yamoe_runs = []
                for _ in range(10):
                    start = time.perf_counter()
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
                    yamoe_runs.append((time.perf_counter() - start) * 1e3)

                yamoe_time = sum(yamoe_runs) / len(yamoe_runs)
                yamoe_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Yamoe: OOM - skipping this batch size")
                    yamoe_success = False
                else:
                    raise e

            # Test reference model
            ref_success = True
            ref_time = None
            ref_mem = None

            try:
                # Setup reference model
                config_obj = type("Config", (), {})()
                config_obj.hidden_size = hidden_dim
                config_obj.intermediate_size = 4 * hidden_dim
                config_obj.num_local_experts = num_experts

                model = reference.GptOssExperts(config_obj)
                model.gate_up_proj.data = gate_up_proj
                model.gate_up_proj_bias.data = gate_up_proj_bias
                model.down_proj.data = down_proj
                model.down_proj_bias.data = down_proj_bias
                model = model.cuda().half()
                model.eval()

                # Warmup runs for reference
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(hidden_states, router_indices, routing_weights)

                # Time and measure memory for reference model
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                ref_runs = []
                with torch.no_grad():
                    for _ in range(10):
                        start = time.perf_counter()
                        ref_output = model(
                            hidden_states, router_indices, routing_weights
                        )
                        torch.cuda.synchronize()
                        ref_runs.append((time.perf_counter() - start) * 1e3)

                ref_time = sum(ref_runs) / len(ref_runs)
                ref_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Reference: OOM - skipping this batch size")
                    ref_success = False
                else:
                    raise e

            # Report results if both succeeded
            if yamoe_success and ref_success:
                yamoe_times.append(yamoe_time)
                yamoe_memory.append(yamoe_mem)
                reference_times.append(ref_time)
                reference_memory.append(ref_mem)
                speedup = ref_time / yamoe_time
                speedups.append(speedup)

                throughput_yamoe = (
                    (batch_size * seq_len * hidden_dim) / (yamoe_time / 1000) / 1e9
                )  # GFLOPS
                throughput_ref = (
                    (batch_size * seq_len * hidden_dim) / (ref_time / 1000) / 1e9
                )  # GFLOPS

                print(
                    f"  Yamoe: {yamoe_time:.3f} ms / {yamoe_mem:.1f} MB / {throughput_yamoe:.2f} GFLOPS"
                )
                print(
                    f"  Reference: {ref_time:.3f} ms / {ref_mem:.1f} MB / {throughput_ref:.2f} GFLOPS"
                )
                print(
                    f"  Speedup: {speedup:.2f}x, Memory reduction: {ref_mem / yamoe_mem:.2f}x, "
                    f"Efficiency gain: {throughput_yamoe / throughput_ref:.2f}x"
                )
            elif yamoe_success and not ref_success:
                # Only Yamoe succeeded - still record its results
                yamoe_times.append(yamoe_time)
                yamoe_memory.append(yamoe_mem)
                # Use None/placeholder values for reference
                reference_times.append(None)
                reference_memory.append(None)
                speedups.append(None)

                throughput_yamoe = (
                    (batch_size * seq_len * hidden_dim) / (yamoe_time / 1000) / 1e9
                )
                print(
                    f"  Yamoe: {yamoe_time:.3f} ms / {yamoe_mem:.1f} MB / {throughput_yamoe:.2f} GFLOPS"
                )
                print(f"  Reference: OOM - unable to measure")
                print(f"  Yamoe runs successfully while Reference OOMs")
            elif not yamoe_success and ref_success:
                # Only Reference succeeded
                yamoe_times.append(None)
                yamoe_memory.append(None)
                reference_times.append(ref_time)
                reference_memory.append(ref_mem)
                speedups.append(None)

                throughput_ref = (
                    (batch_size * seq_len * hidden_dim) / (ref_time / 1000) / 1e9
                )
                print(f"  Yamoe: OOM - unable to measure")
                print(
                    f"  Reference: {ref_time:.3f} ms / {ref_mem:.1f} MB / {throughput_ref:.2f} GFLOPS"
                )
                print(f"  Reference runs successfully while Yamoe OOMs")
            else:
                # Both failed
                yamoe_times.append(None)
                yamoe_memory.append(None)
                reference_times.append(None)
                reference_memory.append(None)
                speedups.append(None)
                print(f"  Both implementations OOM at batch_size={batch_size}")

        except Exception as e:
            print(f"  Unexpected error at batch_size={batch_size}: {str(e)}")
            # Add None values to maintain list consistency
            yamoe_times.append(None)
            yamoe_memory.append(None)
            reference_times.append(None)
            reference_memory.append(None)
            speedups.append(None)

        # Clear GPU memory after each batch size test
        torch.cuda.empty_cache()

    all_results.append(
        {
            "config": config,
            "yamoe_times": yamoe_times,
            "reference_times": reference_times,
            "yamoe_memory": yamoe_memory,
            "reference_memory": reference_memory,
            "speedups": speedups,
        }
    )

# Create comprehensive visualization with time and memory
fig = plt.figure(figsize=(24, 16))

# Create 3 rows: time comparison, memory comparison, combined metrics
for config_idx, result in enumerate(all_results[:6]):
    # Time comparison subplot
    ax1 = plt.subplot(3, 6, config_idx + 1)
    x = np.arange(len(batch_sizes))
    width = 0.35

    # Filter out None values for plotting
    yamoe_times_filtered = [t if t is not None else 0 for t in result["yamoe_times"]]
    ref_times_filtered = [t if t is not None else 0 for t in result["reference_times"]]

    bars1 = ax1.bar(
        x - width / 2,
        yamoe_times_filtered,
        width,
        label="Yamoe",
        color="#1f77b4",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        ref_times_filtered,
        width,
        label="Reference",
        color="#ff7f0e",
        alpha=0.8,
    )

    # Add speedup annotations (only where both values exist)
    for i, (y_time, r_time) in enumerate(
        zip(result["yamoe_times"], result["reference_times"])
    ):
        if y_time is not None and r_time is not None:
            speedup = r_time / y_time
            ax1.text(
                i,
                max(y_time, r_time) * 1.05,
                f"{speedup:.1f}x",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="green",
            )
        elif y_time is not None and r_time is None:
            ax1.text(
                i,
                y_time * 1.05,
                "Y-OK",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="blue",
            )
        elif y_time is None and r_time is not None:
            ax1.text(
                i,
                r_time * 1.05,
                "R-OK",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="orange",
            )
        else:
            ax1.text(
                i,
                0.1,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="red",
            )

    ax1.set_ylabel("Time (ms)", fontsize=9)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes, fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    config = result["config"]
    ax1.set_title(
        f"Time: seq={config['seq_len']}, h={config['hidden_dim']}, e={config['num_experts']}",
        fontsize=8,
        fontweight="bold",
    )

    if config_idx == 0:
        ax1.legend(loc="upper left", fontsize=8)

    # Memory comparison subplot
    ax2 = plt.subplot(3, 6, config_idx + 7)

    # Filter out None values for memory plotting
    yamoe_mem_filtered = [m if m is not None else 0 for m in result["yamoe_memory"]]
    ref_mem_filtered = [m if m is not None else 0 for m in result["reference_memory"]]

    bars3 = ax2.bar(
        x - width / 2,
        yamoe_mem_filtered,
        width,
        label="Yamoe",
        color="#2ca02c",
        alpha=0.8,
    )
    bars4 = ax2.bar(
        x + width / 2,
        ref_mem_filtered,
        width,
        label="Reference",
        color="#d62728",
        alpha=0.8,
    )

    # Add memory reduction annotations (only where both values exist)
    for i, (y_mem, r_mem) in enumerate(
        zip(result["yamoe_memory"], result["reference_memory"])
    ):
        if y_mem is not None and r_mem is not None:
            reduction = r_mem / y_mem
            ax2.text(
                i,
                max(y_mem, r_mem) * 1.05,
                f"{reduction:.1f}x",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="purple",
            )

    ax2.set_ylabel("Memory (MB)", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes, fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title(
        f"Memory: seq={config['seq_len']}, h={config['hidden_dim']}, e={config['num_experts']}",
        fontsize=8,
        fontweight="bold",
    )

    if config_idx == 0:
        ax2.legend(loc="upper left", fontsize=8)

    # Combined speedup and memory efficiency subplot
    ax3 = plt.subplot(3, 6, config_idx + 13)

    # Calculate speedups and memory reductions, handling None values
    valid_speedups = []
    valid_mem_reductions = []
    valid_batch_sizes_speedup = []
    valid_batch_sizes_mem = []

    for i, (r, y) in enumerate(zip(result["reference_times"], result["yamoe_times"])):
        if r is not None and y is not None:
            valid_speedups.append(r / y)
            valid_batch_sizes_speedup.append(batch_sizes[i])

    for i, (r, y) in enumerate(zip(result["reference_memory"], result["yamoe_memory"])):
        if r is not None and y is not None:
            valid_mem_reductions.append(r / y)
            valid_batch_sizes_mem.append(batch_sizes[i])

    if valid_speedups:
        ax3.plot(
            valid_batch_sizes_speedup,
            valid_speedups,
            "o-",
            label="Time Speedup",
            color="green",
            linewidth=2,
            markersize=6,
        )
    if valid_mem_reductions:
        ax3.plot(
            valid_batch_sizes_mem,
            valid_mem_reductions,
            "s-",
            label="Memory Reduction",
            color="purple",
            linewidth=2,
            markersize=6,
        )

    ax3.set_xlabel("Batch Size", fontsize=9)
    ax3.set_ylabel("Improvement Factor", fontsize=9)
    ax3.set_xticks(batch_sizes)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax3.set_title(
        f"Improvements: seq={config['seq_len']}, h={config['hidden_dim']}",
        fontsize=8,
        fontweight="bold",
    )

    if config_idx == 0:
        ax3.legend(loc="upper left", fontsize=8)

plt.suptitle(
    "MoE Performance & Memory Comparison - Yamoe vs Reference",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
plt.savefig("moe_performance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Removed heatmap section per user request

# Print detailed summary
print("\n" + "=" * 80)
print("DETAILED SUMMARY")
print("=" * 80)

for idx, result in enumerate(all_results[:6]):
    config = result["config"]
    print(f"\nConfiguration {idx + 1}:")
    print(
        f"  Parameters: seq_len={config['seq_len']}, hidden_dim={config['hidden_dim']}, "
        f"experts={config['num_experts']}, top_k={config['top_k']}"
    )
    # Handle None values in speedups
    valid_speedups = [s for s in result["speedups"] if s is not None]
    if valid_speedups:
        print(f"  Average Speedup: {sum(valid_speedups) / len(valid_speedups):.2f}x")
        max_speedup = max(valid_speedups)
        min_speedup = min(valid_speedups)
        max_idx = result["speedups"].index(max_speedup)
        min_idx = result["speedups"].index(min_speedup)
        print(f"  Max Speedup: {max_speedup:.2f}x at batch_size={batch_sizes[max_idx]}")
        print(f"  Min Speedup: {min_speedup:.2f}x at batch_size={batch_sizes[min_idx]}")
    else:
        print("  No valid speedup measurements (all OOM)")
