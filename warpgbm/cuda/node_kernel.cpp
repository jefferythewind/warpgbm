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

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &gradients,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_bin_column_kernel(
    at::Tensor X,
    at::Tensor bin_edges,
    at::Tensor bin_indices);

void build_histograms(
    const at::Tensor &bin_indices,
    const at::Tensor &sample_to_node,
    const at::Tensor &residual,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int threads_per_block = 64,
    int rows_per_thread = 4);

void find_splits_for_level(
    const at::Tensor &node_ids, // [N] global node ids
    const at::Tensor &G,        // [N x F x B]
    const at::Tensor &H,        // [N x F x B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &split_gains, // [max_nodes x F]
    at::Tensor &split_bins,  // [max_nodes x F]
    int threads_per_block = 256);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram", &launch_histogram_kernel_cuda, "Histogram (CUDA)");
    m.def("compute_histogram2", &launch_histogram_kernel_cuda_2, "Histogram (CUDA) 2");
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("build_histograms", &build_histograms, "Histogram Builder CUDA");
    m.def("find_splits_for_level", &find_splits_for_level, "Split Finder CUDA");
}