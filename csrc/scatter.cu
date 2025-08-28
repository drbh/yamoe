// csrc/scatter.cu

#include <cstdint>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <type_traits>

// Minimal atomic add shim:
// - native CUDA atomics for float/double
// - 16-bit CAS fallback for Half/BFloat16 (works on all SMs)

// CAS-based 16-bit atomic add (for c10::Half / c10::BFloat16)
template <typename T>
__device__ inline void atomicAdd16(
    T *addr,
    T val) {
  // Find containing 32-bit word and whether we're the high or low 16 bits
  std::uintptr_t uaddr = reinterpret_cast<std::uintptr_t>(addr);
  unsigned int *base =
      reinterpret_cast<unsigned int *>(uaddr & ~std::uintptr_t(0x3));
  const bool hi_half = (uaddr & 0x2) != 0;

  unsigned int old32 = *base, assumed;
  do {
    assumed = old32;

    // Extract current 16-bit payload
    unsigned short cur16 = hi_half ? (assumed >> 16) : (assumed & 0xFFFFu);

    // Reinterpret those 16 bits as T, then promote to float
    T cur;
    *reinterpret_cast<unsigned short *>(&cur) = cur16;
    float f = static_cast<float>(cur) + static_cast<float>(val);

    // Convert back to T (rounds appropriately), grab its 16-bit payload
    T res = static_cast<T>(f);
    unsigned short res16 = *reinterpret_cast<unsigned short *>(&res);

    // Merge back into the correct half and attempt CAS
    unsigned int new32 =
        hi_half ? ((assumed & 0x0000FFFFu) |
                   (static_cast<unsigned int>(res16) << 16))
                : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(res16));

    old32 = atomicCAS(base, assumed, new32);
  } while (old32 != assumed);
}

// Unified atomicAdd for all scalar_t
template <typename T>
__device__ inline void atomicAddT(
    T *addr,
    T val) {
  if constexpr (std::is_same<T, float>::value) {
    atomicAdd(addr, val);
  } else if constexpr (std::is_same<T, double>::value) {
    atomicAdd(addr, val);
  } else {
    // c10::Half or c10::BFloat16
    atomicAdd16(addr, val);
  }
}

// Kernel: y[tok, :] += src[e, i, :] for valid (e,i)
// where tok = indices[bins[e-1] + i] / top_k
template <typename scalar_t>
__global__ void scatter_kernel(
    const scalar_t *__restrict__ src,     // [E, C, H]
    const int *__restrict__ idx,          // [S]
    const int *__restrict__ bins,         // [E] cumulative
    const scalar_t *__restrict__ weights, // [S] routing weights (can be null)
    scalar_t *__restrict__ y,             // [T, H] (accumulated)
    int T,
    int H,
    int E,
    int C,
    int top_k) {
  int e = blockIdx.x;
  int i = blockIdx.y;
  if (e >= E || i >= C)
    return;

  const int end = bins[e];
  const int start = (e == 0) ? 0 : bins[e - 1];
  const int n = end - start;

  bool valid = (i < n);
  int tok = 0;
  if (valid) {
    int flat = idx[start + i];
    tok = flat / top_k;
    if (tok < 0 || tok >= T)
      valid = false; // guard
  }
  if (!valid)
    return;

  const scalar_t *src_row = src + ((size_t)e * C + i) * H;
  scalar_t *y_row = y + (size_t)tok * H;

  // Get the weight/scale factor for this token if provided
  scalar_t scale = (weights != nullptr) ? weights[start + i] : scalar_t(1.0);

  int t = threadIdx.x;
  for (int h = t; h < H; h += blockDim.x) {
    atomicAddT(&y_row[h], src_row[h] * scale);
  }
}

void scatter_cuda(
    const torch::Tensor &src,     // [E, C, H]
    const torch::Tensor &indices, // [S]  (int32)
    const torch::Tensor &bins,    // [E]  cumulative (int32)
    const torch::Tensor &weights, // [S]  routing weights (optional)
    torch::Tensor &y,             // [T, H] (accumulate into)
    int64_t T,                    // tokens
    int64_t E,                    // experts
    int64_t C,                    // capacity
    int64_t top_k                 // router top-k
) {
  const int64_t H = src.size(2);

  // Grid over experts x capacity; threads over H
  dim3 grid(E, C);
  int threads = 256;

  // Include Half + BFloat16 in dispatch
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      src.scalar_type(),
      "scatter_cuda",
      ([&] {
        using scalar_t_ = scalar_t;
        scatter_kernel<scalar_t_><<<grid, threads>>>(
            src.data_ptr<scalar_t_>(),
            indices.data_ptr<int>(),
            bins.data_ptr<int>(),
            weights.defined() ? weights.data_ptr<scalar_t_>() : nullptr,
            y.data_ptr<scalar_t_>(),
            static_cast<int>(T),
            static_cast<int>(H),
            static_cast<int>(E),
            static_cast<int>(C),
            static_cast<int>(top_k));
      }));
}
