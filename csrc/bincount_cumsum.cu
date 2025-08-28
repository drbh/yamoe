// csrc/bincount_cumsum.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

template <typename scalar_t>
__global__ void bincount_cumsum_kernel(
    const scalar_t *__restrict__ input,
    int32_t *__restrict__ bins_out,
    const int n_input,
    const int n_bins) {
  // Shared memory for local bincount
  extern __shared__ int shared_counts[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int threads_per_block = blockDim.x;

  // Initialize shared memory
  for (int i = tid; i < n_bins; i += threads_per_block) {
    shared_counts[i] = 0;
  }
  __syncthreads();

  // Each block processes a chunk of input
  int start = bid * threads_per_block;
  int end = min(start + threads_per_block, n_input);

  // Bincount phase - each thread processes its elements
  for (int i = start + tid; i < end; i += threads_per_block) {
    if (i < n_input) {
      int bin = static_cast<int>(input[i]);
      if (bin >= 0 && bin < n_bins) {
        atomicAdd(&shared_counts[bin], 1);
      }
    }
  }
  __syncthreads();

  // Write block results to global memory
  for (int i = tid; i < n_bins; i += threads_per_block) {
    atomicAdd(&bins_out[i], shared_counts[i]);
  }
  __syncthreads();

  // Only first block does the cumsum
  if (bid == 0) {
    // Simple cumsum on first block
    if (tid == 0) {
      for (int i = 1; i < n_bins; i++) {
        bins_out[i] += bins_out[i - 1];
      }
    }
  }
}

void bincount_cumsum_cuda(
    torch::Tensor input,
    torch::Tensor &bins_out,
    int64_t minlength) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be int32");
  TORCH_CHECK(bins_out.is_cuda(), "Output must be CUDA tensor");

  const auto n_input = input.numel();
  const auto n_bins = static_cast<int>(minlength);

  // Validate output tensor dimensions and clear it
  TORCH_CHECK(bins_out.numel() >= n_bins,
              "Output tensor must have at least minlength elements");
  bins_out.zero_();

  const int threads_per_block = 256;
  const int n_blocks = (n_input + threads_per_block - 1) / threads_per_block;

  // Launch kernel with shared memory for bincount
  const size_t shared_mem_size = n_bins * sizeof(int);

  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(),
      "bincount_cumsum_cuda",
      ([&] {
        bincount_cumsum_kernel<scalar_t>
            <<<n_blocks, threads_per_block, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                bins_out.data_ptr<int32_t>(),
                n_input,
                n_bins);
      }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel failed: ",
              cudaGetErrorString(err));

  // No return needed - output is modified in-place
}