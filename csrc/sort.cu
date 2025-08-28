// csrc/sort.cu
// originally from
// https://github.com/databricks/megablocks/blob/main/csrc/sort.h

#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <torch/torch.h>

#define CUDA_CALL(code)                                                        \
  do {                                                                         \
    cudaError_t status = (code);                                               \
    std::string err = cudaGetErrorString(status);                              \
    TORCH_CHECK(status == cudaSuccess, err);                                   \
  } while (0)

template <typename T>
void cub_radix_sort(
    torch::Tensor x,
    int64_t end_bit,
    torch::Tensor x_out,
    torch::Tensor iota_out) {
  // Get iota for values in sort.
  auto iota_options =
      torch::TensorOptions().dtype(x.scalar_type()).device(x.device());
  torch::Tensor iota = torch::arange(0, x.numel(), iota_options);

  // Get temporary buffer size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      /*d_temp_storage*/ nullptr,
      /*temp_storage_bytes*/ scratchpad_bytes,
      /*d_keys_in*/ x.data_ptr<T>(),
      /*d_keys_out*/ x_out.data_ptr<T>(),
      /*d_values_in*/ iota.data_ptr<T>(),
      /*d_values_out*/ iota_out.data_ptr<T>(),
      /*num_items*/ x.numel(),
      /*begin_bit*/ 0,
      /*end_bit*/ end_bit,
      /*stream*/ c10::cuda::getCurrentCUDAStream()));

  // Allocate scratchpad.
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
  torch::Tensor scratchpad =
      torch::empty(static_cast<long>(scratchpad_bytes), options);

  // Run the kernel.
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      /*d_temp_storage*/ scratchpad.data_ptr(),
      /*temp_storage_bytes*/ scratchpad_bytes,
      /*d_keys_in*/ x.data_ptr<T>(),
      /*d_keys_out*/ x_out.data_ptr<T>(),
      /*d_values_in*/ iota.data_ptr<T>(),
      /*d_values_out*/ iota_out.data_ptr<T>(),
      /*num_items*/ x.numel(),
      /*begin_bit*/ 0,
      /*end_bit*/ end_bit,
      /*stream*/ c10::cuda::getCurrentCUDAStream()));
}

void sort_cuda(
    torch::Tensor x,
    int64_t end_bit,
    torch::Tensor x_out,
    torch::Tensor iota_out) {
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 1);
  TORCH_CHECK(x.scalar_type() == torch::kInt16 ||
              x.scalar_type() == torch::kInt32 ||
              x.scalar_type() == torch::kInt64);
  TORCH_CHECK(x_out.is_cuda());
  TORCH_CHECK(x_out.ndimension() == 1);
  TORCH_CHECK(x_out.scalar_type() == x.scalar_type());
  TORCH_CHECK(iota_out.is_cuda());
  TORCH_CHECK(iota_out.ndimension() == 1);
  TORCH_CHECK(iota_out.scalar_type() == x.scalar_type());

  // Exit early if there is no work to do.
  if (x_out.numel() == 0)
    return;

  switch (x.scalar_type()) {
  case torch::kInt16:
    return cub_radix_sort<short>(x, end_bit, x_out, iota_out);
  case torch::kInt32:
    return cub_radix_sort<int>(x, end_bit, x_out, iota_out);
  default:
    TORCH_CHECK(x.scalar_type() == torch::kInt64);
    return cub_radix_sort<long>(x, end_bit, x_out, iota_out);
  }
}

#undef CUDA_CALL
