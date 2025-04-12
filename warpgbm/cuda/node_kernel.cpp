#include <torch/extension.h>
#include <vector>

// Declare the function from histogram_kernel.cu
void launch_histogram_kernel_cuda(
    const at::Tensor &bin_indices,
    const at::Tensor &gradients,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_kernel_cuda_2(
    const at::Tensor &bin_indices, // int8 [N, F]
    const at::Tensor &gradients,   // float32 [N]
    at::Tensor &grad_hist,         // float32 [F * B]
    at::Tensor &hess_hist,         // float32 [F * B]
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_best_split_kernel_cuda(
    const at::Tensor &G,
    const at::Tensor &H,
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &out_feature,
    at::Tensor &out_bin);

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &gradients,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram", &launch_histogram_kernel_cuda, "Histogram (CUDA)");
    m.def("compute_histogram2", &launch_histogram_kernel_cuda_2, "Histogram (CUDA) 2");
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_best_split_kernel_cuda, "Best Split (CUDA)");
}