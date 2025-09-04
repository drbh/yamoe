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
- availability: easy to install and use via the [kernels](https://github.com/huggingface/kernels) library

### Kernel Hub

You can find the kernel on [Kernel Hub](https://huggingface.co/drbh/yamoe) and install it via the [kernels](https://github.com/huggingface/kernels) library.

```python
from kernels import get_kernel
yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")
```

### Performance

`yamoe` scales well as batch sizes increase in comparision to the naive method of repeating the data and computation for every item in the batch as shown in the reference in [torch-ext/yamoe/reference.py](torch-ext/yamoe/reference.py). This bench can be reproduced by running `uv run perf_plot.py` or a smaller bench and correctness comparision can be run with `uv run compare_example.py`


TLDR: smaller is better on the first two rows of charts

<img width="3583" height="2358" alt="moe_performance_comparison" src="https://github.com/user-attachments/assets/72938f64-ec05-4eaa-82c4-507a43891543" />



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
from kernels import get_local_kernel
from kernels import get_kernel
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

print(f"Output: sum={output.sum().item():.1f}, min={output.min().item():.1f}, max={output.max().item():.1f}")
print(f"First 3: {output.view(-1)[:3].tolist()}")
print(f"Time: {elapsed_ms:.1f}ms, Memory: {peak_mem_mb:.0f}MB")
```
