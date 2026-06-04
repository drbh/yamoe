// csrc/batch_mm.cu

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
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

// Variable-sized ("grouped") batched matmul: for each expert e, computes
//   output[e, :counts[e]] = x[e, :counts[e]] @ weights[e]
// i.e. a separate row-major GEMM [counts[e], H] x [H, N] -> [counts[e], N],
// skipping the padded rows entirely. This is a single cuBLAS launch via
// cublasGemmGroupedBatchedEx (each expert is its own group with M = counts[e]).
//
// Compared to a dense bmm over the full padded capacity, this does work
// proportional to the *actual* per-expert token counts, which is a large win
// when routing is imbalanced (many experts well below capacity).
//
// `counts` is a HOST vector of length E with counts[e] <= x.size(1).
// x, weights and output must be contiguous.
void grouped_mm(
    torch::Tensor x,                 // [E, C, H]
    torch::Tensor weights,           // [E, H, N]
    const std::vector<int> &counts,  // host, length E
    torch::Tensor output) {          // [E, C, N], pre-allocated
  TORCH_CHECK(x.is_cuda() && weights.is_cuda() && output.is_cuda(),
              "grouped_mm tensors must be on CUDA");
  TORCH_CHECK(x.is_contiguous() && weights.is_contiguous() &&
                  output.is_contiguous(),
              "grouped_mm tensors must be contiguous");
  TORCH_CHECK(x.scalar_type() == weights.scalar_type() &&
                  x.scalar_type() == output.scalar_type(),
              "grouped_mm tensors must share dtype");

  const int E = (int)x.size(0);
  const int C = (int)x.size(1);
  const int H = (int)x.size(2);
  const int N = (int)weights.size(2);
  TORCH_CHECK((int)weights.size(1) == H, "weights inner dim must match x");
  TORCH_CHECK((int)output.size(1) == C && (int)output.size(2) == N,
              "output must be [E, C, N]");
  TORCH_CHECK((int)counts.size() == E, "counts must have length E");

  cudaDataType_t cdt;
  switch (x.scalar_type()) {
  case torch::kBFloat16: cdt = CUDA_R_16BF; break;
  case torch::kHalf:     cdt = CUDA_R_16F;  break;
  case torch::kFloat:    cdt = CUDA_R_32F;  break;
  default: TORCH_CHECK(false, "grouped_mm: unsupported dtype");
  }

  const size_t esz = x.element_size();
  const char *xb = static_cast<const char *>(x.data_ptr());
  const char *wb = static_cast<const char *>(weights.data_ptr());
  char *ob = static_cast<char *>(output.data_ptr());

  // One group per (nonzero) expert. We compute the row-major product
  // O[m,N] = X[m,H] @ W[H,N] via the "swap operands" trick so column-major
  // cuBLAS yields the row-major result: cuBLAS m=N, n=count, k=H, with
  // A=W (lda=N), B=X (ldb=H), C=O (ldc=N), both ops non-transposed.
  std::vector<cublasOperation_t> transa, transb;
  std::vector<int> m_arr, n_arr, k_arr, lda, ldb, ldc, group_size;
  std::vector<const void *> Aarr, Barr;
  std::vector<void *> Carr;
  transa.reserve(E); transb.reserve(E);
  m_arr.reserve(E); n_arr.reserve(E); k_arr.reserve(E);
  lda.reserve(E); ldb.reserve(E); ldc.reserve(E);
  Aarr.reserve(E); Barr.reserve(E); Carr.reserve(E); group_size.reserve(E);

  for (int e = 0; e < E; ++e) {
    const int m = counts[e];
    if (m <= 0) continue; // expert received no tokens
    TORCH_CHECK(m <= C, "counts[e] exceeds capacity");
    transa.push_back(CUBLAS_OP_N);
    transb.push_back(CUBLAS_OP_N);
    m_arr.push_back(N);    // cuBLAS m
    n_arr.push_back(m);    // cuBLAS n (this expert's token count)
    k_arr.push_back(H);    // cuBLAS k
    lda.push_back(N);      // A = W[e]  (col-major [N, H])
    ldb.push_back(H);      // B = X[e]  (col-major [H, count])
    ldc.push_back(N);      // C = O[e]  (col-major [N, count])
    Aarr.push_back(wb + (size_t)e * H * N * esz);
    Barr.push_back(xb + (size_t)e * C * H * esz);
    Carr.push_back(ob + (size_t)e * C * N * esz);
    group_size.push_back(1);
  }

  const int group_count = (int)group_size.size();
  if (group_count == 0)
    return; // no tokens at all

  // alpha/beta: one scalar per group, in the scale (compute) type (fp32).
  std::vector<float> alpha(group_count, 1.0f), beta(group_count, 0.0f);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasStatus_t status = cublasGemmGroupedBatchedEx(
      handle,
      transa.data(), transb.data(),
      m_arr.data(), n_arr.data(), k_arr.data(),
      alpha.data(),
      Aarr.data(), cdt, lda.data(),
      Barr.data(), cdt, ldb.data(),
      beta.data(),
      Carr.data(), cdt, ldc.data(),
      group_count, group_size.data(),
      CUBLAS_COMPUTE_32F);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "cublasGemmGroupedBatchedEx failed with status ", (int)status);

  // cublasGemmGroupedBatchedEx consumes the host parameter arrays (dims,
  // pointers, ld) asynchronously on the stream, so they must outlive the launch.
  // Sync the stream before these local vectors go out of scope and are freed.
  cudaError_t serr =
      cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(serr == cudaSuccess,
              "grouped_mm stream sync failed: ", cudaGetErrorString(serr));
}
