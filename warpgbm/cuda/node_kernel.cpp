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

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

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

torch::Tensor h_des_mc(
    torch::Tensor bin_idx,
    torch::Tensor era_of_row,
    torch::Tensor G,
    torch::Tensor H,
    torch::Tensor feat_idx,
    torch::Tensor idx_mat,
    torch::Tensor idx_len,
    torch::Tensor era_ends,
    int B,
    bool enable_compact,
    bool root_fastpath);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_directional_split_kernel, "Best Split (CUDA)");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
    // ---------------- PyBind ----------------
    m.def("h_des_mc", &h_des_mc,
          "Multiclass hist (butterfly compact + class-batched butterfly hist) -> [2,E,k,K,B]");

  
}