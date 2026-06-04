// csrc/experts_gather.cu
//
// Decode-optimized "fused gather" MoE forward.
//
// For T tokens each routed to K experts there are N = T*K (token, expert) pairs.
// Instead of running all E experts (dense) or sorting tokens into per-expert bins
// (the grouped path), this reads ONLY the active experts' weights, directly from
// the weight tensor via device pointer arrays passed to cublasGemmBatchedEx. The
// "gather" is pure pointer indirection -- no index_select copy, no sort, no host
// sync -- so it is also CUDA-graph capturable.
//
// It wins when N is small (decode / tiny batch): it touches ~N expert matrices
// instead of all E. For large T it re-reads shared experts once per pair and
// loses to the grouped path; use it only for small token counts.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <type_traits>

// --- 16-bit atomic add shim (CAS), mirrors scatter.cu ---------------------
template <typename T>
__device__ inline void atomicAdd16(T *addr, T val) {
  std::uintptr_t uaddr = reinterpret_cast<std::uintptr_t>(addr);
  unsigned int *base =
      reinterpret_cast<unsigned int *>(uaddr & ~std::uintptr_t(0x3));
  const bool hi_half = (uaddr & 0x2) != 0;
  unsigned int old32 = *base, assumed;
  do {
    assumed = old32;
    unsigned short cur16 = hi_half ? (assumed >> 16) : (assumed & 0xFFFFu);
    T cur;
    *reinterpret_cast<unsigned short *>(&cur) = cur16;
    float f = static_cast<float>(cur) + static_cast<float>(val);
    T res = static_cast<T>(f);
    unsigned short res16 = *reinterpret_cast<unsigned short *>(&res);
    unsigned int new32 =
        hi_half ? ((assumed & 0x0000FFFFu) |
                   (static_cast<unsigned int>(res16) << 16))
                : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(res16));
    old32 = atomicCAS(base, assumed, new32);
  } while (old32 != assumed);
}

template <typename T> __device__ inline void atomicAddT(T *addr, T val) {
  if constexpr (std::is_same<T, float>::value)
    atomicAdd(addr, val);
  else if constexpr (std::is_same<T, double>::value)
    atomicAdd(addr, val);
  else
    atomicAdd16(addr, val);
}

// Build the per-batch pointer arrays for cublasGemmBatchedEx entirely on-device.
//   A[n] = Abase + A_idx[n]*Astride   (weight matrix for this pair's expert)
//   B[n] = Bbase + B_idx[n]*Bstride   (input row: token for gate_up, pair for down)
//   C[n] = Cbase + n*Cstride          (output row for this pair)
__global__ void build_ptrs_kernel(const int *__restrict__ A_idx,
                                  const int *__restrict__ B_idx,
                                  const char *Abase, long Astride,
                                  const char *Bbase, long Bstride,
                                  char *Cbase, long Cstride,
                                  const void **A, const void **B, void **C,
                                  int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  A[n] = reinterpret_cast<const void *>(Abase + (long)A_idx[n] * Astride);
  B[n] = reinterpret_cast<const void *>(Bbase + (long)B_idx[n] * Bstride);
  C[n] = reinterpret_cast<void *>(Cbase + (long)n * Cstride);
}

// Fused bias-add + GLU over the flat [N, 2H] gate_up rows. Bias is indexed by the
// pair's expert id. One block per pair.
template <typename scalar_t>
__global__ void glu_flat_kernel(const scalar_t *__restrict__ gate_up, // [N, 2H]
                                const int *__restrict__ expert_ids,    // [N]
                                const scalar_t *__restrict__ bias, // [E, 2H]
                                scalar_t *__restrict__ out,        // [N, H]
                                int H) {
  const int n = blockIdx.x;
  const int e = expert_ids[n];
  const size_t gu = (size_t)n * 2 * H;
  const size_t bo = (size_t)e * 2 * H;
  const size_t oo = (size_t)n * H;
  const float limit = 7.0f, alpha = 1.702f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float g = (float)gate_up[gu + 2 * h] + (float)bias[bo + 2 * h];
    float u = (float)gate_up[gu + 2 * h + 1] + (float)bias[bo + 2 * h + 1];
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    const float glu = g / (1.0f + expf(-alpha * g));
    out[oo + h] = (scalar_t)((u + 1.0f) * glu);
  }
}

// Add the expert's down bias, scale by the routing weight, and accumulate into
// the token's output row. One block per pair; K pairs accumulate into each token.
template <typename scalar_t>
__global__ void
scatter_weighted_kernel(const scalar_t *__restrict__ down_out,  // [N, H]
                        const int *__restrict__ expert_ids,     // [N]
                        const scalar_t *__restrict__ down_bias, // [E, H]
                        const scalar_t *__restrict__ rweights,  // [T, E]
                        scalar_t *__restrict__ y,               // [T, H]
                        int H, int E, int K) {
  const int n = blockIdx.x;
  const int t = n / K;
  const int e = expert_ids[n];
  const float w = (float)rweights[(size_t)t * E + e];
  const size_t do_off = (size_t)n * H;
  const size_t b_off = (size_t)e * H;
  const size_t y_off = (size_t)t * H;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float v = ((float)down_out[do_off + h] + (float)down_bias[b_off + h]) * w;
    atomicAddT(&y[y_off + h], (scalar_t)v);
  }
}

static cudaDataType_t cuda_dtype(const torch::Tensor &t) {
  switch (t.scalar_type()) {
  case torch::kBFloat16: return CUDA_R_16BF;
  case torch::kHalf:     return CUDA_R_16F;
  case torch::kFloat:    return CUDA_R_32F;
  default: TORCH_CHECK(false, "experts_gather: unsupported dtype"); return CUDA_R_32F;
  }
}

torch::Tensor experts_gather_cuda(
    torch::Tensor hidden_states,     // [T, H]
    torch::Tensor router_indices,    // [T, K]
    torch::Tensor routing_weights,   // [T, E] (dense)
    torch::Tensor gate_up_proj,      // [E, H, 2H]
    torch::Tensor gate_up_proj_bias, // [E, 2H]
    torch::Tensor down_proj,         // [E, H, H]
    torch::Tensor down_proj_bias,    // [E, H]
    int64_t num_experts,             // E
    int64_t top_k                    // K
) {
  TORCH_CHECK(hidden_states.is_cuda() && gate_up_proj.is_cuda(),
              "experts_gather: tensors must be on CUDA");
  TORCH_CHECK(hidden_states.ndimension() == 2, "hidden_states must be [T, H]");

  hidden_states = hidden_states.contiguous();
  gate_up_proj = gate_up_proj.contiguous();
  gate_up_proj_bias = gate_up_proj_bias.contiguous();
  down_proj = down_proj.contiguous();
  down_proj_bias = down_proj_bias.contiguous();
  routing_weights = routing_weights.contiguous();

  const int64_t T = hidden_states.size(0);
  const int64_t H = hidden_states.size(1);
  const int64_t E = num_experts;
  const int64_t K = top_k;
  const int64_t N = T * K;
  TORCH_CHECK(routing_weights.size(0) == T && routing_weights.size(1) == E,
              "routing_weights must be dense [T, E]");
  TORCH_CHECK(gate_up_proj.size(0) == E && gate_up_proj.size(1) == H &&
                  gate_up_proj.size(2) == 2 * H,
              "gate_up_proj must be [E, H, 2H]");
  TORCH_CHECK(down_proj.size(0) == E && down_proj.size(1) == H &&
                  down_proj.size(2) == H,
              "down_proj must be [E, H, H]");

  auto stream = c10::cuda::getCurrentCUDAStream();
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device());
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device());
  auto fopt = torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device());

  // Per-pair index tensors (device): expert id, source token, and pair id.
  torch::Tensor expert_ids = router_indices.reshape({N}).to(torch::kInt32).contiguous();
  torch::Tensor iota = torch::arange(N, i32);
  torch::Tensor token_ids = torch::floor_divide(iota, (int64_t)K).to(torch::kInt32);

  // Working buffers.
  torch::Tensor gate_up = torch::empty({N, 2 * H}, fopt);
  torch::Tensor glu = torch::empty({N, H}, fopt);
  torch::Tensor down_out = torch::empty({N, H}, fopt);
  torch::Tensor output = torch::zeros({T, H}, fopt);

  // Device pointer arrays for cublasGemmBatchedEx (reused for both GEMMs).
  torch::Tensor Ap = torch::empty({N}, i64);
  torch::Tensor Bp = torch::empty({N}, i64);
  torch::Tensor Cp = torch::empty({N}, i64);
  auto A = reinterpret_cast<const void **>(Ap.data_ptr<int64_t>());
  auto B = reinterpret_cast<const void **>(Bp.data_ptr<int64_t>());
  auto C = reinterpret_cast<void **>(Cp.data_ptr<int64_t>());

  const size_t esz = hidden_states.element_size();
  const cudaDataType_t dt = cuda_dtype(hidden_states);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.0f, beta = 0.0f;
  const int threads = (int)std::min<int64_t>(H, 1024);
  const int blk = 256;
  const int pblocks = (int)((N + blk - 1) / blk);

  // --- Gate/up projection: out[1, 2H] = x[1, H] @ W_gu[H, 2H], per pair -----
  // Pointers: A = gate_up_proj[e] (by expert), B = hidden[t] (by token).
  build_ptrs_kernel<<<pblocks, blk, 0, stream>>>(
      expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(),
      (const char *)gate_up_proj.data_ptr(), (long)(H * 2 * H * esz),
      (const char *)hidden_states.data_ptr(), (long)(H * esz),
      (char *)gate_up.data_ptr(), (long)(2 * H * esz), A, B, C, (int)N);
  // Col-major "swap operands" trick: cuBLAS m=2H, n=1, k=H gives the row-major
  // result O[1, 2H] = X[1, H] @ W[H, 2H] (A=W lda=2H, B=X ldb=H, C=O ldc=2H).
  TORCH_CHECK(cublasGemmBatchedEx(
                  handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(2 * H), 1, (int)H,
                  &alpha, A, dt, (int)(2 * H), B, dt, (int)H, &beta, C, dt,
                  (int)(2 * H), (int)N, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
              "experts_gather: gate_up batched GEMM failed");

  // --- Fused bias + GLU: [N, 2H] -> [N, H] ----------------------------------
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, hidden_states.scalar_type(), "glu_flat", ([&] {
        glu_flat_kernel<scalar_t><<<(int)N, threads, 0, stream>>>(
            gate_up.data_ptr<scalar_t>(), expert_ids.data_ptr<int>(),
            gate_up_proj_bias.data_ptr<scalar_t>(), glu.data_ptr<scalar_t>(),
            (int)H);
      }));

  // --- Down projection: out[1, H] = glu[1, H] @ W_dn[H, H], per pair --------
  // Pointers: A = down_proj[e] (by expert), B = glu[n] (by pair).
  build_ptrs_kernel<<<pblocks, blk, 0, stream>>>(
      expert_ids.data_ptr<int>(), iota.data_ptr<int>(),
      (const char *)down_proj.data_ptr(), (long)(H * H * esz),
      (const char *)glu.data_ptr(), (long)(H * esz),
      (char *)down_out.data_ptr(), (long)(H * esz), A, B, C, (int)N);
  TORCH_CHECK(cublasGemmBatchedEx(
                  handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)H, 1, (int)H, &alpha, A,
                  dt, (int)H, B, dt, (int)H, &beta, C, dt, (int)H, (int)N,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) ==
                  CUBLAS_STATUS_SUCCESS,
              "experts_gather: down batched GEMM failed");

  // --- Down bias + routing weight + scatter-add into [T, H] -----------------
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, hidden_states.scalar_type(), "scatter_weighted",
      ([&] {
        scatter_weighted_kernel<scalar_t><<<(int)N, threads, 0, stream>>>(
            down_out.data_ptr<scalar_t>(), expert_ids.data_ptr<int>(),
            down_proj_bias.data_ptr<scalar_t>(),
            routing_weights.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            (int)H, (int)E, (int)K);
      }));

  return output;
}
