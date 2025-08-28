#pragma once

#include <torch/torch.h>

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

torch::Tensor
batch_mm(torch::Tensor x,           // [E, C, H] - expert tokens
         torch::Tensor weights,     // [E, H, H_out] - expert weight matrices
         torch::Tensor batch_sizes, // [E] - actual tokens per expert (<=C)
         torch::Tensor output,      // [E, C, H_out] - output buffer
         bool trans_b = false       // transpose weights if needed
);

torch::Tensor experts_cuda(
    torch::Tensor hidden_states,     // [T, H] - flattened hidden states
    torch::Tensor router_indices,    // [T, K] - expert indices per token
    torch::Tensor routing_weights,   // [T, E] or [T, K] - routing weights
    torch::Tensor gate_up_proj,      // [E, H, 2*H] - gate/up projection weights
    torch::Tensor gate_up_proj_bias, // [E, 2*H] - gate/up projection bias
    torch::Tensor down_proj,         // [E, H, H] - down projection weights
    torch::Tensor down_proj_bias,    // [E, H] - down projection bias
    int64_t expert_capacity,         // C - capacity per expert
    int64_t num_experts,             // E - number of experts
    int64_t top_k                    // K - top-k routing
);
