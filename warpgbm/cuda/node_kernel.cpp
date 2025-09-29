#include <torch/extension.h>
#include <vector>

// Declare the function from histogram_kernel.cu

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       // [E, F, B]
    at::Tensor &per_era_direction,  // [E, F, B]
    int threads = 128);

void launch_histogram_gh_fm_warpagg(
        const at::Tensor& bin_fm,          // int8  [F_master, N]
        const at::Tensor& grad_vec,        // float [N]
        const at::Tensor& hess_vec,        // float [N]
        const at::Tensor& sample_indices,  // int32 [R]
        const at::Tensor& feature_indices, // int32 [F_sub]
        const at::Tensor& era_indices,     // int32 [N]
        at::Tensor& grad_hist,             // float [E, F_sub, B]
        at::Tensor& hess_hist,             // float [E, F_sub, B]
        int num_bins,
        int threads_per_block = 256,
        int rows_per_thread = 4);

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

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram3", &launch_histogram_gh_fm_warpagg, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_directional_split_kernel, "Best Split (CUDA)");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
}