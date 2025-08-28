---
license: mit
tags:
  - kernel
---

``` 
                                                            
oooo    ooo  .oooo.   ooo. .oo.  .oo.    .ooooo.   .ooooo.  
 `88.  .8'  `P  )88b  `888P"Y88bP"Y88b  d88' `88b d88' `88b 
  `88..8'    .oP"888   888   888   888  888   888 888ooo888 
   `888'    d8(  888   888   888   888  888   888 888    .o 
    .8'     `Y888""8o o888o o888o o888o `Y8bod8P' `Y8bod8P' 
.o..P'                                                      
`Y8P'                                                       

                Yet Another Mixture of Experts 
```

`yamoe` is a no nonsense, straightforward implementation of Mixture of Experts (MoE) kernels, designed to be super easy to use and be very computationally efficient.

### Design goals
- simplicity: easy to read and understand the code
- efficiency: optimized for high throughput and low latency
- low memory usage: optimized to handle large batch sizes
- reproducibility: easy to reproduce results, no special new `sm` requirements


### How to use

```python
# /// script
# requires-python = "==3.10"
# dependencies = ["torch==2.7.0", "triton", "numpy", "kernels"]
# [tool.uv.sources]
# kernels = { git = "https://github.com/huggingface/kernels.git" }
# ///

import time
import torch
from kernels import get_kernel
from pathlib import Path
from torch.nn import functional as F

yamoe = get_kernel("drbh/yamoe")

# Configuration
torch.manual_seed(0)
batch_size, seq_len, hidden_dim = 128, 2048, 2880
num_experts, top_k = 32, 4

# Create routing weights
logits = torch.randn(batch_size, seq_len, num_experts)
probs = F.softmax(logits, dim=-1)
weights, indices = torch.topk(probs, top_k, dim=-1)

batch_seq = batch_size * seq_len
routing_weights = torch.zeros(batch_seq, num_experts, dtype=weights.dtype)
flat_indices, flat_weights = indices.reshape(-1, top_k), weights.reshape(-1, top_k)
batch_indices = torch.arange(batch_seq).unsqueeze(1).expand(-1, top_k)
routing_weights[batch_indices, flat_indices] = flat_weights

# Create model tensors (scaled to prevent overflow)
hidden_states = torch.randn(batch_size, seq_len, hidden_dim).cuda().half() * 0.1
gate_up_proj = torch.randn(num_experts, hidden_dim, 2 * hidden_dim).cuda().half() * 0.02
gate_up_proj_bias = torch.zeros(num_experts, 2 * hidden_dim).cuda().half()
down_proj = torch.randn(num_experts, hidden_dim, hidden_dim).cuda().half() * 0.02
down_proj_bias = torch.zeros(num_experts, hidden_dim).cuda().half()
routing_weights = routing_weights.cuda().half()
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

print(f"Output sum: {output.sum().item():.4f}")
print(f"Kernel time: {elapsed_ms:.3f} ms")
print(f"Peak GPU memory: {peak_mem_mb:.2f} MB")
# Output sum: 124.2500
# Kernel time: 85.722 ms
# Peak GPU memory: 8403.40 MB

```