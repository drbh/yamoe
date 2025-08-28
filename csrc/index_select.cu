// csrc/index_select.cu

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

template <typename scalar_t>
__global__ void index_select_kernel(
    const scalar_t *__restrict__ in,
    const int32_t *__restrict__ idx,
    scalar_t *__restrict__ out,
    int64_t N) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = in[(int64_t)idx[i]];
}

torch::Tensor index_select_out_cuda(
    torch::Tensor out,       // [N], same dtype/device as in
    torch::Tensor in,        // [M], contiguous
    torch::Tensor idx_int32) // [N], int32, contiguous
{
  TORCH_CHECK(in.is_cuda() && idx_int32.is_cuda() && out.is_cuda(),
              "cuda only");
  TORCH_CHECK(idx_int32.dtype() == torch::kInt32, "idx must be int32");
  TORCH_CHECK(
      in.is_contiguous() && idx_int32.is_contiguous() && out.is_contiguous(),
      "contiguous required");

  int64_t N = idx_int32.numel();
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kBFloat16,
      torch::kHalf,
      in.scalar_type(),
      "index_select_int32",
      [&] {
        const scalar_t *pin = in.data_ptr<scalar_t>();
        const int32_t *pidx = idx_int32.data_ptr<int32_t>();
        scalar_t *pout = out.data_ptr<scalar_t>();
        index_select_kernel<scalar_t>
            <<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(pin,
                                                                        pidx,
                                                                        pout,
                                                                        N);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
