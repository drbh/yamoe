#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(
    TORCH_EXTENSION_NAME,
    ops) {
  ops.def("gather("
          "Tensor x, "
          "Tensor indices, "
          "Tensor bins, "
          "Tensor! output, "
          "int E, "
          "int C, "
          "int top_k) -> ()");
  ops.impl("gather", torch::kCUDA, &gather_cuda);

  ops.def("scatter("
          "Tensor src, "
          "Tensor indices, "
          "Tensor bins, "
          "Tensor weights, "
          "Tensor! y, "
          "int T, "
          "int E, "
          "int C, "
          "int top_k) -> ()");
  ops.impl("scatter", torch::kCUDA, &scatter_cuda);

  ops.def("sort("
          "Tensor x, "
          "int end_bit, "
          "Tensor! x_out, "
          "Tensor! iota_out) -> ()");
  ops.impl("sort", torch::kCUDA, &sort_cuda);

  ops.def("bincount_cumsum("
          "Tensor input, "
          "Tensor! output, "
          "int minlength) -> ()");
  ops.impl("bincount_cumsum", torch::kCUDA, &bincount_cumsum_cuda);

  ops.def("index_select_out("
          "Tensor! out, "
          "Tensor input, "
          "Tensor idx_int32) -> Tensor");
  ops.impl("index_select_out", torch::kCUDA, &index_select_out_cuda);

  ops.def("batch_mm("
          "Tensor x, "
          "Tensor weights, "
          "Tensor batch_sizes, "
          "Tensor! output, "
          "bool trans_b=False) -> Tensor");
  ops.impl("batch_mm", torch::kCUDA, &batch_mm);

  ops.def("experts("
          "Tensor hidden_states, "
          "Tensor router_indices, "
          "Tensor routing_weights, "
          "Tensor gate_up_proj, "
          "Tensor gate_up_proj_bias, "
          "Tensor down_proj, "
          "Tensor down_proj_bias, "
          "int expert_capacity, "
          "int num_experts, "
          "int top_k) -> Tensor");
  ops.impl("experts", torch::kCUDA, &experts_cuda);

  ops.def("experts_backward("
          "Tensor grad_out, "
          "Tensor hidden_states, "
          "Tensor router_indices, "
          "Tensor routing_weights, "
          "Tensor gate_up_proj, "
          "Tensor gate_up_proj_bias, "
          "Tensor down_proj, "
          "Tensor down_proj_bias, "
          "int expert_capacity, "
          "int num_experts, "
          "int top_k) -> Tensor[]");
  ops.impl("experts_backward", torch::kCUDA, &experts_backward_cuda);
}

REGISTER_EXTENSION(
    TORCH_EXTENSION_NAME)