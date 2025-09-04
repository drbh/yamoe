// Backward pass for MoE experts

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

void sort_cuda(torch::Tensor x,
               int64_t end_bit,
               torch::Tensor x_out,
               torch::Tensor iota_out);
void bincount_cumsum_cuda(torch::Tensor input,
                          torch::Tensor &output,
                          int64_t minlength);
void gather_cuda(const torch::Tensor &x,
                 const torch::Tensor &indices,
                 const torch::Tensor &bins,
                 torch::Tensor &output,
                 int64_t E,
                 int64_t C,
                 int64_t top_k);
void scatter_cuda(const torch::Tensor &src,
                  const torch::Tensor &indices,
                  const torch::Tensor &bins,
                  const torch::Tensor &weights,
                  torch::Tensor &y,
                  int64_t T,
                  int64_t E,
                  int64_t C,
                  int64_t top_k);
torch::Tensor index_select_out_cuda(torch::Tensor out,
                                    torch::Tensor in,
                                    torch::Tensor idx_int32);

// scatter gradients back to expert outputs and routing weights
template <typename scalar_t>
__global__ void binned_scatter_backward_kernel(
    const scalar_t *__restrict__ grad_y,           // [T, H]
    const int *__restrict__ indices,               // [S]
    const int *__restrict__ bins,                  // [E+1]
    const scalar_t *__restrict__ selected_weights, // [S]
    const scalar_t *__restrict__ expert_output,    // [E, C, H]
    scalar_t *__restrict__ grad_expert_output,     // [E, C, H]
    scalar_t *__restrict__ grad_selected_weights,  // [S]
    int T,
    int K,
    int H,
    int E,
    int C) {

  int e = blockIdx.x;
  int i = blockIdx.y;
  if (e >= E || i >= C)
    return;

  const int start = (e == 0) ? 0 : bins[e - 1];
  const int end = bins[e];
  const int n_all = end - start;
  const int take = (n_all > 0) ? min(n_all, C) : 0;

  if (take == 0 || i >= take) {
    scalar_t *dst = grad_expert_output + ((size_t)e * C + i) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x)
      dst[h] = scalar_t(0);
    return;
  }

  const int sorted_pos = start + i;
  const int flat_pos = indices[sorted_pos];
  const int tok = flat_pos / K;

  const scalar_t scale = selected_weights[sorted_pos];

  const scalar_t *grad_y_ptr = grad_y + (size_t)tok * H;
  scalar_t *grad_exp_ptr = grad_expert_output + ((size_t)e * C + i) * H;
  const scalar_t *expert_ptr = expert_output + ((size_t)e * C + i) * H;

  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    grad_exp_ptr[h] += grad_y_ptr[h] * scale;
  }

  if (threadIdx.x == 0) {
    scalar_t sum = scalar_t(0);
    for (int h = 0; h < H; ++h)
      sum += grad_y_ptr[h] * expert_ptr[h];
    gpuAtomicAdd(&grad_selected_weights[flat_pos], sum);
  }
}

// gather gradients back to hidden states
template <typename scalar_t>
__global__ void binned_gather_backward_kernel(
    const scalar_t *__restrict__ grad_x, // [E, C, H]
    const int *__restrict__ indices,     // [S]
    const int *__restrict__ bins,        // [E+1]
    scalar_t *__restrict__ grad_hidden,  // [T, H]
    int T,
    int K,
    int H,
    int E,
    int C) {

  int e = blockIdx.x;
  int i = blockIdx.y;
  if (e >= E || i >= C)
    return;

  const int start = (e == 0) ? 0 : bins[e - 1];
  const int end = bins[e];
  const int n = min(max(end - start, 0), C);
  if (i >= n)
    return;

  const int flat_pos = indices[start + i];
  const int tok = flat_pos / K;

  const scalar_t *gx = grad_x + ((size_t)e * C + i) * H;
  scalar_t *gh = grad_hidden + (size_t)tok * H;

  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    gpuAtomicAdd(&gh[h], gx[h]);
  }
}

std::vector<torch::Tensor> experts_backward_cuda(
    const torch::Tensor &grad_out,
    const torch::Tensor &hidden_states,
    const torch::Tensor &router_indices,
    const torch::Tensor &routing_weights,
    const torch::Tensor &gate_up_proj,
    const torch::Tensor &gate_up_proj_bias,
    const torch::Tensor &down_proj,
    const torch::Tensor &down_proj_bias,
    int64_t expert_capacity,
    int64_t num_experts,
    int64_t top_k) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
  TORCH_CHECK(router_indices.is_cuda(), "router_indices must be CUDA");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be CUDA");
  TORCH_CHECK(gate_up_proj.is_cuda() && down_proj.is_cuda(),
              "weights must be CUDA");
  TORCH_CHECK(gate_up_proj_bias.is_cuda() && down_proj_bias.is_cuda(),
              "biases must be CUDA");

  const at::cuda::OptionalCUDAGuard device_guard(grad_out.device());

  const int64_t T = hidden_states.size(0);
  const int64_t H = hidden_states.size(1);
  const int64_t E = num_experts;
  const int64_t C = expert_capacity;
  const int64_t K = top_k;

  TORCH_CHECK(router_indices.dim() == 2 && router_indices.size(0) == T &&
                  router_indices.size(1) == K,
              "router_indices must be [T, K]");

  auto float_opts = hidden_states.options();
  auto i32_opts = torch::TensorOptions()
                      .device(hidden_states.device())
                      .dtype(torch::kInt32);

  // Sort tokens by expert ID
  torch::Tensor flat_indices =
      router_indices.contiguous().view({-1}).to(torch::kInt32);
  torch::Tensor sorted_values = torch::empty_like(flat_indices);
  torch::Tensor sorted_indices = torch::empty_like(flat_indices);
  sort_cuda(flat_indices, 32, sorted_values, sorted_indices);

  // Compute expert boundaries
  torch::Tensor bins = torch::empty({E + 1}, i32_opts);
  bincount_cumsum_cuda(sorted_values, bins, E);
  cudaDeviceSynchronize();

  // Gather tokens for each expert
  torch::Tensor x = torch::empty({E, C, H}, float_opts);
  gather_cuda(hidden_states.contiguous(), sorted_indices, bins, x, E, C, K);

  // Gate-up projection
  torch::Tensor gate_up = at::bmm(x.contiguous(), gate_up_proj.contiguous());
  gate_up.add_(gate_up_proj_bias.unsqueeze(1));

  // GLU activation (recompute forward)
  auto gu_pair = gate_up.view({E, C, H, 2});
  torch::Tensor pre_gate = gu_pair.select(3, 0);
  torch::Tensor pre_up = gu_pair.select(3, 1);

  const double limit = 7.0;
  const double alpha = 1.702;
  torch::Tensor gate_clamped = at::clamp_max(pre_gate, limit);
  torch::Tensor up_clamped = at::clamp(pre_up, -limit, limit);
  torch::Tensor s = at::sigmoid(gate_clamped * alpha);
  torch::Tensor gate_act = gate_clamped * s;
  torch::Tensor up_out = (1 + up_clamped) * gate_act;

  // Down projection
  torch::Tensor y_expert = at::bmm(up_out.contiguous(), down_proj.contiguous());
  y_expert.add_(down_proj_bias.unsqueeze(1));

  // Get routing weights in sorted order
  torch::Tensor flat_router = router_indices.view({T, K});
  torch::Tensor selected_2d;
  if (routing_weights.size(1) == K) {
    selected_2d = routing_weights.contiguous();
  } else {
    TORCH_CHECK(routing_weights.size(1) == E,
                "routing_weights must be [T,K] or [T,E]");
    selected_2d = at::gather(routing_weights, 1, flat_router.to(torch::kLong));
  }
  torch::Tensor selected_flat = selected_2d.contiguous().view({T * K});
  torch::Tensor weights_sorted = torch::empty_like(selected_flat);
  index_select_out_cuda(weights_sorted, selected_flat, sorted_indices);

  // Initialize gradients
  torch::Tensor dHidden = torch::zeros_like(hidden_states);
  torch::Tensor dRouting;
  torch::Tensor dWgu = torch::zeros_like(gate_up_proj);
  torch::Tensor dbgu = torch::zeros_like(gate_up_proj_bias);
  torch::Tensor dWd = torch::zeros_like(down_proj);
  torch::Tensor dbd = torch::zeros_like(down_proj_bias);

  // Reshape grad_out to [T,H]
  TORCH_CHECK(grad_out.numel() == T * H || grad_out.numel() == T * K * H,
              "grad_out numel must be T*H or T*K*H");
  torch::Tensor grad_y = (grad_out.numel() == T * H)
                             ? grad_out.contiguous().view({T, H})
                             : grad_out.contiguous().view({T, K, H}).sum(1);

  // Backward through scatter
  torch::Tensor grad_expert_output = torch::zeros({E, C, H}, float_opts);
  torch::Tensor grad_selected_weights = torch::zeros({T * K}, float_opts);
  {
    dim3 grid(E, C);
    int threads = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        hidden_states.scalar_type(),
        "binned_scatter_backward",
        [&] {
          using st = scalar_t;
          binned_scatter_backward_kernel<st>
              <<<grid, threads>>>(grad_y.data_ptr<st>(),
                                  sorted_indices.data_ptr<int>(),
                                  bins.data_ptr<int>(),
                                  weights_sorted.data_ptr<st>(),
                                  y_expert.data_ptr<st>(),
                                  grad_expert_output.data_ptr<st>(),
                                  grad_selected_weights.data_ptr<st>(),
                                  (int)T,
                                  (int)K,
                                  (int)H,
                                  (int)E,
                                  (int)C);
        });
    cudaDeviceSynchronize();
  }

  // Route weight gradients
  torch::Tensor grad_selected_flat = torch::zeros({T * K}, float_opts);
  grad_selected_flat.index_add_(0,
                                sorted_indices.to(torch::kLong),
                                grad_selected_weights);

  if (routing_weights.size(1) == E) {
    torch::Tensor flat_grad_routing = torch::zeros_like(routing_weights);
    flat_grad_routing.scatter_add_(1,
                                   flat_router.to(torch::kLong),
                                   grad_selected_flat.view({T, K}));
    dRouting = flat_grad_routing;
  } else {
    dRouting = grad_selected_flat.view({T, K});
  }

  // Backward through down projection
  dbd = grad_expert_output.sum(1);
  torch::Tensor grad_intermediate =
      torch::bmm(grad_expert_output.contiguous(),
                 down_proj.transpose(1, 2).contiguous());
  dWd = torch::bmm(up_out.transpose(1, 2).contiguous(),
                   grad_expert_output.contiguous());

  // Backward through GLU
  torch::Tensor grad_up_plus_1 = grad_intermediate * gate_act;
  torch::Tensor grad_glu = grad_intermediate * (up_clamped + 1);
  torch::Tensor grad_up_clamped = grad_up_plus_1;

  torch::Tensor sigmoid_gate = torch::sigmoid(gate_clamped * alpha);
  torch::Tensor grad_gate_clamped =
      grad_glu *
      (sigmoid_gate + gate_clamped * sigmoid_gate * (1 - sigmoid_gate) * alpha);

  // Unclamp gradients
  torch::Tensor grad_gate = grad_gate_clamped.clone();
  grad_gate.masked_fill_(pre_gate > limit, 0);
  torch::Tensor grad_up = grad_up_clamped.clone();
  grad_up.masked_fill_(pre_up > limit, 0);
  grad_up.masked_fill_(pre_up < -limit, 0);

  // Merge gate/up gradients
  torch::Tensor grad_gate_up_pair = torch::zeros({E, C, H, 2}, float_opts);
  grad_gate_up_pair.select(3, 0).copy_(grad_gate);
  grad_gate_up_pair.select(3, 1).copy_(grad_up);
  torch::Tensor grad_gate_up = grad_gate_up_pair.view({E, C, 2 * H});

  // Backward through gate-up projection
  dbgu = grad_gate_up.sum(1);
  torch::Tensor grad_x = torch::bmm(grad_gate_up.contiguous(),
                                    gate_up_proj.transpose(1, 2).contiguous());
  dWgu = torch::bmm(x.transpose(1, 2).contiguous(), grad_gate_up.contiguous());

  // Backward through gather
  torch::Tensor grad_hidden = torch::zeros({T, H}, float_opts);
  {
    dim3 grid(E, C);
    int threads = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
                                    at::kBFloat16,
                                    hidden_states.scalar_type(),
                                    "binned_gather_backward",
                                    [&] {
                                      using st = scalar_t;
                                      binned_gather_backward_kernel<st>
                                          <<<grid, threads>>>(
                                              grad_x.data_ptr<st>(),
                                              sorted_indices.data_ptr<int>(),
                                              bins.data_ptr<int>(),
                                              grad_hidden.data_ptr<st>(),
                                              (int)T,
                                              (int)K,
                                              (int)H,
                                              (int)E,
                                              (int)C);
                                    });
    cudaDeviceSynchronize();
  }
  dHidden += grad_hidden;

  return {dHidden, dRouting, dWgu, dbgu, dWd, dbd};
}
