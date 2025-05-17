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
    const at::Tensor &G, // [F x B]
    const at::Tensor &H, // [F x B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &best_gains, // [F], float32
    at::Tensor &best_bins,
    int threads);

void launch_histogram_tiled_era(
    const at::Tensor &bin_indices,  // int8
    const at::Tensor &era_ids,      // int16
    const at::Tensor &gradients,    // float
    at::Tensor &grad_hist,          // float, shape [F, B*E]
    at::Tensor &hess_hist,          // float, shape [F, B*E]
    int             B,
    int             E,
    int             threads_per_block = 256,
    int             rows_per_thread   = 4);

void launch_bin_column_kernel(
    at::Tensor X,
    at::Tensor bin_edges,
    at::Tensor bin_indices);

void predict_with_forest(
    const at::Tensor &bin_indices, // [N x F], int8
    const at::Tensor &tree_tensor, // [T x max_nodes x 6], float32
    float learning_rate,
    at::Tensor &out // [N], float32
);

void launch_tie_breaker(
    const at::Tensor& G,    // [F, B, E]
    const at::Tensor& H,    // [F, B, E]
    const at::Tensor& tie_feats, // [K]
    float min_gain, float min_child, float eps,
    at::Tensor& out_gain,  // [K]
    at::Tensor& out_bin,   // [K]
    int threads_per_block
);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram", &launch_histogram_kernel_cuda, "Histogram (CUDA)");
    m.def("compute_histogram2", &launch_histogram_kernel_cuda_2, "Histogram (CUDA) 2");
    m.def("compute_histogram3", &launch_histogram_tiled_era, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_best_split_kernel_cuda, "Best Split (CUDA)");
    m.def("tie_breaker", &launch_tie_breaker, "Best Split (CUDA)");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
}