// csrc/moe.cpp

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

// Forward declarations for existing functions
void sort_cuda(torch::Tensor x,
               int64_t end_bit,
               torch::Tensor x_out,
               torch::Tensor iota_out);

void bincount_cumsum_cuda(torch::Tensor input,
                          torch::Tensor &output,
                          int64_t minlength);

torch::Tensor index_select_out_cuda(torch::Tensor out,
                                    torch::Tensor in,
                                    torch::Tensor idx_int32);

void gather_cuda(torch::Tensor const &x,
                 torch::Tensor const &indices,
                 torch::Tensor const &bins,
                 torch::Tensor &output,
                 int64_t E,
                 int64_t C,
                 int64_t top_k);

void scatter_cuda(torch::Tensor const &src,
                  torch::Tensor const &indices,
                  torch::Tensor const &bins,
                  torch::Tensor const &weights,
                  torch::Tensor &y,
                  int64_t T,
                  int64_t E,
                  int64_t C,
                  int64_t top_k);

torch::Tensor batch_mm(torch::Tensor x,
                       torch::Tensor weights,
                       torch::Tensor batch_sizes,
                       torch::Tensor output,
                       bool trans_b = false);

torch::Tensor experts_cuda(
    torch::Tensor hidden_states,     // [B*S, H] - flattened hidden states
    torch::Tensor router_indices,    // [B*S, K] - expert indices per token
    torch::Tensor routing_weights,   // [B*S, E] or [B*S, K] - routing weights
    torch::Tensor gate_up_proj,      // [E, H, 2*H] - gate/up projection weights
    torch::Tensor gate_up_proj_bias, // [E, 2*H] - gate/up projection bias
    torch::Tensor down_proj,         // [E, H, H] - down projection weights
    torch::Tensor down_proj_bias,    // [E, H] - down projection bias
    int64_t expert_capacity,         // C - capacity per expert
    int64_t num_experts,             // E - number of experts
    int64_t top_k                    // K - top-k routing
) {
  // Input validation
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
  TORCH_CHECK(router_indices.is_cuda(), "router_indices must be on CUDA");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be on CUDA");
  TORCH_CHECK(gate_up_proj.is_cuda(), "gate_up_proj must be on CUDA");
  TORCH_CHECK(gate_up_proj_bias.is_cuda(), "gate_up_proj_bias must be on CUDA");
  TORCH_CHECK(down_proj.is_cuda(), "down_proj must be on CUDA");
  TORCH_CHECK(down_proj_bias.is_cuda(), "down_proj_bias must be on CUDA");

  TORCH_CHECK(hidden_states.ndimension() == 2,
              "hidden_states must be 2D [T, H]");
  TORCH_CHECK(router_indices.ndimension() == 2,
              "router_indices must be 2D [T, K]");
  TORCH_CHECK(routing_weights.ndimension() == 2,
              "routing_weights must be 2D [T, K]");
  TORCH_CHECK(gate_up_proj.ndimension() == 3,
              "gate_up_proj must be 3D [E, H, 2*H]");
  TORCH_CHECK(gate_up_proj_bias.ndimension() == 2,
              "gate_up_proj_bias must be 2D [E, 2*H]");
  TORCH_CHECK(down_proj.ndimension() == 3, "down_proj must be 3D [E, H, H]");
  TORCH_CHECK(down_proj_bias.ndimension() == 2,
              "down_proj_bias must be 2D [E, H]");

  const int64_t T = hidden_states.size(0); // Total tokens
  const int64_t H = hidden_states.size(1); // Hidden size
  const int64_t E = num_experts;
  const int64_t C = expert_capacity;
  const int64_t K = top_k;

  TORCH_CHECK(router_indices.size(0) == T && router_indices.size(1) == K);
  TORCH_CHECK(routing_weights.size(0) == T && (routing_weights.size(1) == K ||
                                               routing_weights.size(1) == E),
              "routing_weights must be [T, K] or [T, E]");
  TORCH_CHECK(gate_up_proj.size(0) == E && gate_up_proj.size(1) == H &&
              gate_up_proj.size(2) == 2 * H);
  TORCH_CHECK(gate_up_proj_bias.size(0) == E &&
              gate_up_proj_bias.size(1) == 2 * H);
  TORCH_CHECK(down_proj.size(0) == E && down_proj.size(1) == H &&
              down_proj.size(2) == H);
  TORCH_CHECK(down_proj_bias.size(0) == E && down_proj_bias.size(1) == H);

  // Ensure simple contiguity where helpful
  hidden_states = hidden_states.contiguous();
  router_indices = router_indices.contiguous();
  routing_weights = routing_weights.contiguous();

  // ALLOCATE

  auto device_opts = torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(hidden_states.device());
  auto int64_opts = torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(hidden_states.device());
  auto float_opts = torch::TensorOptions()
                        .dtype(hidden_states.dtype())
                        .device(hidden_states.device());

  // Buffers for sorting
  torch::Tensor flat_indices =
      router_indices.flatten().to(torch::kInt32, /*non_blocking=*/true);
  torch::Tensor sorted_values = torch::empty_like(flat_indices);
  torch::Tensor sorted_indices = torch::empty_like(flat_indices);

  // Buffer for bins - use int32 for smaller footprint
  torch::Tensor bins =
      torch::empty({E + 1},
                   device_opts); // Pre-allocate for bincount_cumsum result

  // Buffer for gathered tokens
  torch::Tensor x = torch::empty({E, C, H}, float_opts);

  // Buffer for expert token counts
  torch::Tensor expert_tokens = torch::empty({E}, device_opts);

  // Buffers for intermediate results
  torch::Tensor gate_up = torch::empty({E, C, 2 * H}, float_opts);

  // Final output buffer
  torch::Tensor output = torch::zeros_like(hidden_states);

  // COMPUTE

  // Sort tokens by expert
  sort_cuda(flat_indices, 32, sorted_values, sorted_indices);

  // Compute bins using bincount_cumsum
  bincount_cumsum_cuda(sorted_values, bins, E);

  // Gather tokens by expert
  // [T, H] -> [E, C, H]
  gather_cuda(hidden_states, sorted_indices, bins, x, E, C, K);

  if (E > 1) {
    expert_tokens.slice(0, 0, E - 1) =
        bins.slice(0, 1, E) - bins.slice(0, 0, E - 1);
    expert_tokens[E - 1] =
        (int32_t)(flat_indices.size(0) - bins[E - 1].item<int32_t>());
  } else {
    expert_tokens[0] = (int32_t)flat_indices.size(0);
  }
  // Clamp to expert capacity
  expert_tokens = torch::clamp(expert_tokens, 0, (int32_t)C);

  batch_mm(x, gate_up_proj, expert_tokens, gate_up, true);

  // add the gate bias to the output in-place
  gate_up.add_(gate_up_proj_bias.unsqueeze(1));

  // Compute GLU in-place, reusing gate_up buffer for output
  auto gate = gate_up.index({torch::indexing::Ellipsis,
                             torch::indexing::Slice(torch::indexing::None,
                                                    torch::indexing::None,
                                                    2)});
  auto up =
      gate_up.index({torch::indexing::Ellipsis,
                     torch::indexing::Slice(1, torch::indexing::None, 2)});

  const float limit = 7.0f;
  gate = gate.clamp(/*min=*/c10::nullopt, /*max=*/limit);
  up = up.clamp(/*min=*/-limit, /*max=*/limit);

  gate.mul_(torch::sigmoid(gate * 1.702f));
  up.add_(1).mul_(gate);

  // Down projection uses GLU result directly
  gate_up.resize_(0);
  batch_mm(up, down_proj, expert_tokens, gate_up, true);

  // add the down_bias in-place
  gate_up.add_(down_proj_bias.unsqueeze(1));

  // Stage allocations right before use
  torch::Tensor selected_weights = torch::empty({T * K}, float_opts);
  torch::Tensor weights_sorted = torch::empty({T * K}, float_opts);

  torch::Tensor selected_weights_2d =
      selected_weights.view({T, K}); // named lvalue view
  torch::Tensor flat_dense = routing_weights.view({T, E});
  torch::Tensor flat_router = router_indices.view({T, K});

  // gather_out(out&, self, dim, index, sparse_grad=false)
  at::gather_out(selected_weights_2d,
                 flat_dense,
                 /*dim=*/1,
                 flat_router,
                 /*sparse_grad=*/false);

  // Use int32 index select to avoid dtype conversion
  index_select_out_cuda(weights_sorted,                 // [T*K], float_opts
                        selected_weights.view({T * K}), // const&, ok as rvalue
                        sorted_indices // int32 indices, no conversion needed
  );

  // Scatter back to original positions with weights applied
  scatter_cuda(gate_up.view({E, C, H}),
               sorted_indices,
               bins,
               weights_sorted,
               output,
               T,
               E,
               C,
               K);

  return output;
}
