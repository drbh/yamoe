// csrc/batch_mm.cu

#include <torch/torch.h>

// Simply use a standard bmm for now but this can be adapted for
// faster batched expert matrix multiply if needed
torch::Tensor batch_mm(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor batch_sizes,
    torch::Tensor output,
    bool trans_b) {
  // Validate inputs
  TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
  TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");
  TORCH_CHECK(batch_sizes.is_cuda(), "batch_sizes must be on CUDA");

  TORCH_CHECK(x.ndimension() == 3, "x must be 3D tensor"); // [E, C, H]
  TORCH_CHECK(weights.ndimension() == 3,
              "weights must be 3D tensor"); // [E, H, H_out]
  TORCH_CHECK(batch_sizes.ndimension() == 1,
              "batch_sizes must be 1D tensor"); // [E]

  TORCH_CHECK(x.size(0) == weights.size(0) && x.size(0) == batch_sizes.size(0));
  TORCH_CHECK(x.size(2) == weights.size(1)); // H dimension match

  // For now, just fall back to bmm to test the binding
  // torch::bmm(x, weights, output);
  torch::bmm_out(output, x, weights);
  return output;
}
