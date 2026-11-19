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

- decode: a GEMV kernel that streams only the active experts' weights, with a matrix-core tile for experts that get several tokens (ROCm)
- prefill: 16-row and 64-row matrix-core grouped GEMM tiles staged through shared memory, MFMA on ROCm and `mma.sync` on CUDA sm80+
- long prefill: `torch.nn.functional.grouped_mm` (hipBLASLt / CUTLASS), earlier on Hopper and Blackwell where CUTLASS is fast

Thresholds and launch configs are per backend in `torch-ext/yamoe/experts.py`. tok/s compared to `experts_implementation="grouped_mm"`, bf16:

Qwen3-30B-A3B on MI300X

| | grouped_mm | yamoe |
|---|---|---|
| prefill 128 / 512 / 2048 / 8192 tokens | 619 / 2233 / 8238 / 20052 | 3350 / 11943 / 22687 / 21763 |
| decode, static cache + compile, batch 1 / 4 / 16 / 64 | 87 / 197 / 276 / 321 | 103 / 322 / 874 / 1709 |
| decode, eager, batch 1 / 16 | 23 / 294 | 26 / 414 |

Qwen3-30B-A3B on B200

| | grouped_mm | yamoe |
|---|---|---|
| prefill 128 tokens (wall, launch bound) | 41.7 ms | 27.6 ms |
| prefill 512+ tokens | same | same |
| decode, static cache + compile, batch 1 / 4 / 16 / 64 | 135 / 290 / 425 / 503 | 150 / 429 / 969 / 1978 |
| decode, eager, batch 1 / 16 | 30 / 392 | 37 / 581 |

OLMoE-1B-7B on RTX 4090

| | grouped_mm | yamoe |
|---|---|---|
| prefill 128 / 512 / 2048 / 8192 tokens | 4963 / 16193 / 31795 / 33351 | 7960 / 24836 / 40516 / 33340 |
| decode, static cache + compile, batch 1 / 4 / 16 | 174 / 171 / 185 | 249 / 582 / 1509 |
| decode, eager, batch 1 / 16 | 124 / 228 | 154 / 1150 |

Tests: `YAMOE_BUILD=/path/to/dir/with/build pytest tests/test_experts.py`
