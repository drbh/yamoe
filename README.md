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
yamoe = get_kernel("drbh/yamoe", revision="main")
```
