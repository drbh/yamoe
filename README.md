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

You can find the kernel on [Kernel Hub](https://huggingface.co/kernels/drbh/yamoe) and install it via the [kernels](https://github.com/huggingface/kernels) library.

```python
from kernels import get_kernel
yamoe = get_kernel("drbh/yamoe", revision=1)
```

### transformers experts (v2, CUDA and ROCm)

v2 adds `fused_experts`, an experts forward for the standard transformers experts layout (`gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`, SiLU gate, no bias). This covers the Qwen MoE family (Qwen2-MoE, Qwen3-MoE, Qwen3-Next, Qwen3.5 / 3.6 MoE, Qwen3-VL MoE), Kimi-Linear and other models that use `@use_experts_implementation` with these settings. It is one AOT-compiled C++ source for both backends, with no JIT, and it is CUDA-graph and `torch.compile` friendly.

```python
import torch
from kernels import get_kernel
from transformers import AutoModelForCausalLM
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

yamoe = get_kernel("drbh/yamoe", version=2)
ALL_EXPERTS_FUNCTIONS.register("yamoe", yamoe.experts_forward)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507", dtype=torch.bfloat16, device_map="cuda", experts_implementation="yamoe"
)
```

Unsupported layouts (bias, transposed or interleaved weights, other activations) fall back to transformers' `grouped_mm` path.

The kernel picks a path from the average number of tokens per expert:

- decode: a GEMV kernel that streams only the active experts' weights, with an MFMA tile on ROCm for experts that get several tokens
- prefill (ROCm): 16-row and 64-row MFMA grouped GEMM tiles staged through LDS
- long prefill, and anything past decode on CUDA: `torch.nn.functional.grouped_mm`

Qwen3-30B-A3B on MI300X, bf16, compared to `experts_implementation="grouped_mm"`:

| | grouped_mm | yamoe |
|---|---|---|
| prefill 128 / 512 / 2048 tokens (tok/s) | 603 / 2266 / 8044 | 3268 / 11393 / 21324 |
| decode, static cache + compile, batch 1 / 4 / 16 / 64 (tok/s) | 87 / 199 / 276 / 312 | 103 / 320 / 858 / 1360 |
| decode, eager, batch 1 / 16 (tok/s) | 23 / 294 | 26 / 396 |

Tests: `YAMOE_BUILD=/path/to/dir/with/build pytest tests/test_experts.py`
