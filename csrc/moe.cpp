// csrc/moe.cpp

#include <algorithm>
#include <vector>

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

// Fused, capacity-aware bias-add + GLU over only the real token rows.
void fused_glu_cuda(torch::Tensor gate_up,
                    torch::Tensor bias,
                    torch::Tensor counts,
                    torch::Tensor out);

// Per-expert bias add over only the real token rows.
void add_bias_rows_cuda(torch::Tensor out,
                        torch::Tensor bias,
                        torch::Tensor counts);

// Variable-sized batched matmul keyed on per-expert host token counts.
void grouped_mm(torch::Tensor x,
                torch::Tensor weights,
                const std::vector<int> &counts,
                torch::Tensor output);

// Shared implementation. `capture_safe` selects between two ways of driving the
// per-expert GEMMs:
//
//   * capture_safe = false (default): pull per-expert token counts to the HOST
//     (one D2H copy of `bins`) and run a variable-sized grouped cuBLAS GEMM that
//     processes only the real tokens. Fastest under imbalanced / loose-capacity
//     routing, but the host read + cublas grouped GEMM's stream sync make it
//     impossible to capture in a CUDA graph.
//
//   * capture_safe = true: compute per-expert counts on the DEVICE (no D2H) and
//     run fixed-capacity batched GEMMs over [E, C, H]. No host sync anywhere, so
//     the whole forward is CUDA-graph capturable. At a tight `expert_capacity`
//     (capacity_factor ~1.0, C = ceil(T*K/E)) the batched GEMM computes ~exactly
//     the routed-token count, i.e. it matches the grouped GEMM's work with zero
//     padding waste. This is the regime (decode / small batch) where graphs and
//     their launch-overhead removal actually matter.
static torch::Tensor experts_impl(
    torch::Tensor hidden_states,     // [B*S, H] - flattened hidden states
    torch::Tensor router_indices,    // [B*S, K] - expert indices per token
    torch::Tensor routing_weights,   // [B*S, E] or [B*S, K] - routing weights
    torch::Tensor gate_up_proj,      // [E, H, 2*H] - gate/up projection weights
    torch::Tensor gate_up_proj_bias, // [E, 2*H] - gate/up projection bias
    torch::Tensor down_proj,         // [E, H, H] - down projection weights
    torch::Tensor down_proj_bias,    // [E, H] - down projection bias
    int64_t expert_capacity,         // C - capacity per expert
    int64_t num_experts,             // E - number of experts
    int64_t top_k,                   // K - top-k routing
    bool capture_safe                // device-driven, CUDA-graph capturable
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

  // Final output buffer
  torch::Tensor output = torch::zeros_like(hidden_states);

  // COMPUTE

  // Sort tokens by expert. Values are expert ids in [0, E), so we only need
  // ceil(log2(E)) radix bits rather than a full 32-bit sort.
  int64_t sort_bits = 1;
  while ((1LL << sort_bits) < E)
    sort_bits++;
  sort_cuda(flat_indices, sort_bits, sorted_values, sorted_indices);

  // Compute bins using bincount_cumsum
  bincount_cumsum_cuda(sorted_values, bins, E);

  // Ensure weights/biases are contiguous for the GEMMs and the fused elementwise
  // kernels (no-op when already so).
  gate_up_proj = gate_up_proj.contiguous();
  down_proj = down_proj.contiguous();
  gate_up_proj_bias = gate_up_proj_bias.contiguous();
  down_proj_bias = down_proj_bias.contiguous();

  // Per-expert token counts derived from the (inclusive prefix-sum) bins the same
  // way gather does: count_e = bins[e] - bins[e-1] (bins[-1] = 0), clamped to the
  // capacity C (this is where token dropping happens).
  std::vector<int> counts;  // host counts (non-capture path only)
  torch::Tensor counts_dev; // device counts for the capacity-aware kernels
  int64_t C_active;         // capacity the working buffers are sized to

  if (!capture_safe) {
    // Pull counts to the host (one D2H copy) to drive the variable-sized grouped
    // GEMMs and to size the active-capacity buffers. Active capacity = the
    // largest per-expert load actually present (<= C), so unused capacity slots
    // cost no compute or memory. Exact: tokens are packed at the front of each
    // expert's bin and anything past C is dropped here.
    torch::Tensor bins_cpu_t = bins.to(torch::kCPU, torch::kInt32);
    const int32_t *bp = bins_cpu_t.data_ptr<int32_t>();
    counts.resize(E);
    C_active = 1;
    int prev = 0;
    for (int e = 0; e < E; ++e) {
      const int n = bp[e] - prev; // raw token count for expert e
      prev = bp[e];
      const int c = std::max(0, std::min<int>(n, (int)C)); // clamp to capacity
      counts[e] = c;
      C_active = std::max<int64_t>(C_active, c);
    }
    counts_dev = torch::from_blob((void *)counts.data(), {E}, torch::kInt32)
                     .to(hidden_states.device());
  } else {
    // Capture-safe: compute counts entirely on-device (no D2H), and size buffers
    // to the host-known padded capacity C (a device-derived C_active would itself
    // require a D2H). counts_dev[e] = clamp(bins[e] - bins[e-1], 0, C).
    C_active = C;
    torch::Tensor bins_e = bins.narrow(0, 0, E);          // [E]: bins[0..E-1]
    torch::Tensor bins_prev = torch::zeros({E}, device_opts); // bins[e-1], bins[-1]=0
    if (E > 1)
      bins_prev.narrow(0, 1, E - 1).copy_(bins.narrow(0, 0, E - 1));
    counts_dev = (bins_e - bins_prev).clamp(0, C).to(torch::kInt32).contiguous();
  }

  // Buffer for gathered tokens: [E, C_active, H]
  torch::Tensor x = torch::empty({E, C_active, H}, float_opts);

  // Buffer for intermediate results: [E, C_active, 2H]
  torch::Tensor gate_up = torch::empty({E, C_active, 2 * H}, float_opts);

  // Gather tokens by expert: [T, H] -> [E, C_active, H]
  gather_cuda(hidden_states, sorted_indices, bins, x, E, C_active, K);

  // Gate/up projection. Non-capture: variable-sized grouped GEMM over only the
  // real tokens (host counts). Capture-safe: fixed-capacity batched GEMM over
  // [E, C, H] @ [E, H, 2H] (no host counts, no stream sync); padded rows are
  // computed but never read (gather fills, and fused_glu/scatter mask via counts).
  if (!capture_safe)
    grouped_mm(x, gate_up_proj, counts, gate_up);
  else
    torch::bmm_out(gate_up, x, gate_up_proj);

  // Fused bias-add + GLU over only the real token rows: [E, C, 2H] -> [E, C, H].
  // This processes ~total tokens of traffic (not E * C) and replaces the
  // separate bias add, de-interleave, clamps, sigmoid and multiply.
  torch::Tensor glu_out = torch::empty({E, C_active, H}, float_opts);
  fused_glu_cuda(gate_up, gate_up_proj_bias, counts_dev, glu_out);

  // Down projection (grouped or fixed-capacity batched GEMM) + per-expert bias.
  torch::Tensor down_out = torch::empty({E, C_active, H}, float_opts);
  if (!capture_safe)
    grouped_mm(glu_out, down_proj, counts, down_out);
  else
    torch::bmm_out(down_out, glu_out, down_proj);
  add_bias_rows_cuda(down_out, down_proj_bias, counts_dev);

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
  scatter_cuda(down_out,
               sorted_indices,
               bins,
               weights_sorted,
               output,
               T,
               E,
               C_active,
               K);

  return output;
}

// Default forward: host-driven variable-sized grouped GEMM (fastest, not
// CUDA-graph capturable).
torch::Tensor experts_cuda(
    torch::Tensor hidden_states,
    torch::Tensor router_indices,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_proj,
    torch::Tensor gate_up_proj_bias,
    torch::Tensor down_proj,
    torch::Tensor down_proj_bias,
    int64_t expert_capacity,
    int64_t num_experts,
    int64_t top_k) {
  return experts_impl(hidden_states, router_indices, routing_weights,
                      gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias,
                      expert_capacity, num_experts, top_k,
                      /*capture_safe=*/false);
}

// Capture-safe forward: device-driven, fixed-capacity batched GEMMs. No host
// sync, so the whole forward can be captured in a CUDA graph. Best paired with a
// tight expert_capacity so the fixed-capacity GEMM matches the routed-token work.
torch::Tensor experts_static_cuda(
    torch::Tensor hidden_states,
    torch::Tensor router_indices,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_proj,
    torch::Tensor gate_up_proj_bias,
    torch::Tensor down_proj,
    torch::Tensor down_proj_bias,
    int64_t expert_capacity,
    int64_t num_experts,
    int64_t top_k) {
  return experts_impl(hidden_states, router_indices, routing_weights,
                      gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias,
                      expert_capacity, num_experts, top_k,
                      /*capture_safe=*/true);
}
