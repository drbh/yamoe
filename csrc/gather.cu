// csrc/gather.cu

#include <cuda_runtime.h>
#include <torch/torch.h>

template <typename scalar_t>
__global__ void gather_kernel(
    const scalar_t *__restrict__ x, // [T,H]
    const int *__restrict__ idx,    // [S]
    const int *__restrict__ bins,   // [E] cumulative
    scalar_t *__restrict__ out,     // [E,C,H]
    int T,
    int H,
    int E,
    int C,
    int top_k) {
  int e = blockIdx.x; // expert
  int i = blockIdx.y; // row within capacity
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

  const scalar_t *src = valid ? (x + (size_t)tok * H) : nullptr;
  scalar_t *dst = out + ((size_t)e * C + i) * H;

  int t = threadIdx.x;

  // Try vectorized 16B moves if H is multiple of 4 and pointers are aligned
  // (only for float type)
  if constexpr (std::is_same<scalar_t, float>::value) {
    if ((H % 4) == 0 && ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0) &&
        (!valid || (reinterpret_cast<uintptr_t>(src) & 0xF) == 0)) {
      const int HV = H / 4;
      using F4 = float4;
      const F4 *src4 = reinterpret_cast<const F4 *>(src);
      F4 *dst4 = reinterpret_cast<F4 *>(dst);

      for (int j = t; j < HV; j += blockDim.x) {
        F4 v;
        if (valid)
          v = src4[j];
        else
          v = make_float4(0.f, 0.f, 0.f, 0.f);
        dst4[j] = v;
      }
      return;
    }
  }

  // Fallback to scalar copy
  for (int j = t; j < H; j += blockDim.x) {
    dst[j] = valid ? src[j] : scalar_t(0);
  }
}

void gather_cuda(
    torch::Tensor const &x,       // [T, H]
    torch::Tensor const &indices, // [S]
    torch::Tensor const &bins,    // [E] cumulative
    torch::Tensor &output,        // [E, C, H] pre-allocated output buffer
    int64_t E,                    // number of experts
    int64_t C,                    // expert capacity
    int64_t top_k                 // top-k value
) {
  // Get dimensions
  int64_t T = x.size(0);
  int64_t H = x.size(1);

  // Validate output tensor dimensions
  TORCH_CHECK(output.size(0) == E && output.size(1) == C && output.size(2) == H,
              "Output tensor must have shape [E, C, H]");

  // Launch kernel with 2D grid (E, C)
  dim3 grid(E, C);
  int threads = 256;

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
                                  at::kBFloat16,
                                  x.scalar_type(),
                                  "gather_cuda",
                                  ([&] {
                                    using scalar_t_ =
                                        scalar_t; // avoid shadowing surprises
                                    gather_kernel<scalar_t_><<<grid, threads>>>(
                                        x.data_ptr<scalar_t_>(),
                                        indices.data_ptr<int>(),
                                        bins.data_ptr<int>(),
                                        output.data_ptr<scalar_t_>(),
                                        (int)T,
                                        (int)H,
                                        (int)E,
                                        (int)C,
                                        (int)top_k);
                                  }));

  // No return needed - output is modified in-place
}