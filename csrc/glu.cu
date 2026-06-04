// csrc/glu.cu
//
// Fused, capacity-aware GLU for the gpt-oss MoE. Operates on the padded
// [E, C, *] expert buffers but processes only the *real* token rows
// (i < counts[e]), so its memory traffic is proportional to the actual token
// load rather than E * C. It also fuses the per-expert bias add, the gate/up
// de-interleave, the clamps, the SiLU-style gate and the final multiply into a
// single pass (one read of 2H, one write of H) instead of ~6 separate
// elementwise kernels over the full padded buffer.

#include <cuda_runtime.h>
#include <torch/torch.h>

template <typename scalar_t>
__global__ void fused_glu_kernel(
    const scalar_t *__restrict__ gate_up, // [E, C, 2H] (interleaved gate/up)
    const scalar_t *__restrict__ bias,    // [E, 2H]
    const int *__restrict__ counts,       // [E]
    scalar_t *__restrict__ out,           // [E, C, H]
    int C,
    int H) {
  const int e = blockIdx.x;
  const int i = blockIdx.y;
  if (i >= counts[e])
    return; // padded row: skip entirely

  const size_t gu_off = ((size_t)e * C + i) * (size_t)(2 * H);
  const size_t b_off = (size_t)e * (size_t)(2 * H);
  const size_t o_off = ((size_t)e * C + i) * (size_t)H;

  const float limit = 7.0f;
  const float alpha = 1.702f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float g = (float)gate_up[gu_off + 2 * h] + (float)bias[b_off + 2 * h];
    float u = (float)gate_up[gu_off + 2 * h + 1] + (float)bias[b_off + 2 * h + 1];
    g = fminf(g, limit);                  // gate: clamp max only
    u = fminf(fmaxf(u, -limit), limit);   // up:   clamp [-limit, limit]
    const float glu = g / (1.0f + expf(-alpha * g)); // g * sigmoid(alpha * g)
    out[o_off + h] = (scalar_t)((u + 1.0f) * glu);
  }
}

void fused_glu_cuda(
    torch::Tensor gate_up, // [E, C, 2H]
    torch::Tensor bias,    // [E, 2H]
    torch::Tensor counts,  // [E] int32
    torch::Tensor out) {   // [E, C, H]
  TORCH_CHECK(gate_up.is_cuda() && bias.is_cuda() && counts.is_cuda() &&
              out.is_cuda());
  TORCH_CHECK(gate_up.is_contiguous() && bias.is_contiguous() &&
              out.is_contiguous());
  const int E = (int)gate_up.size(0);
  const int C = (int)gate_up.size(1);
  const int H = (int)out.size(2);
  TORCH_CHECK((int)gate_up.size(2) == 2 * H, "gate_up must be [E, C, 2H]");
  TORCH_CHECK((int)out.size(0) == E && (int)out.size(1) == C);

  dim3 grid(E, C);
  int threads = H < 1024 ? H : 1024;
  if (threads < 1)
    threads = 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, gate_up.scalar_type(), "fused_glu", ([&] {
        fused_glu_kernel<scalar_t><<<grid, threads>>>(
            gate_up.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            counts.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            C,
            H);
      }));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_glu launch failed");
}

// Per-expert bias add over only the real token rows (i < counts[e]).
template <typename scalar_t>
__global__ void add_bias_rows_kernel(
    scalar_t *__restrict__ out,        // [E, C, H]
    const scalar_t *__restrict__ bias, // [E, H]
    const int *__restrict__ counts,    // [E]
    int C,
    int H) {
  const int e = blockIdx.x;
  const int i = blockIdx.y;
  if (i >= counts[e])
    return;
  const size_t o_off = ((size_t)e * C + i) * (size_t)H;
  const size_t b_off = (size_t)e * (size_t)H;
  for (int h = threadIdx.x; h < H; h += blockDim.x)
    out[o_off + h] = (scalar_t)((float)out[o_off + h] + (float)bias[b_off + h]);
}

void add_bias_rows_cuda(
    torch::Tensor out,    // [E, C, H]
    torch::Tensor bias,   // [E, H]
    torch::Tensor counts) {
  TORCH_CHECK(out.is_cuda() && bias.is_cuda() && counts.is_cuda());
  TORCH_CHECK(out.is_contiguous() && bias.is_contiguous());
  const int E = (int)out.size(0);
  const int C = (int)out.size(1);
  const int H = (int)out.size(2);
  dim3 grid(E, C);
  int threads = H < 1024 ? H : 1024;
  if (threads < 1)
    threads = 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, out.scalar_type(), "add_bias_rows", ([&] {
        add_bias_rows_kernel<scalar_t><<<grid, threads>>>(
            out.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            counts.data_ptr<int>(),
            C,
            H);
      }));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_bias_rows launch failed");
}
