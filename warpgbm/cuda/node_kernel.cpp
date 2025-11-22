#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

// Forward declarations
void launch_directional_split_kernel(
    const at::Tensor &G, 
    const at::Tensor &H, 
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       
    at::Tensor &per_era_direction,  
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
    const at::Tensor &bin_indices, 
    const at::Tensor &tree_tensor, 
    float learning_rate,
    at::Tensor &out 
);

// Updated signature
std::vector<torch::Tensor> h_des_mc(
    torch::Tensor bin_indices,   
    torch::Tensor grads,         
    torch::Tensor hess,          
    torch::Tensor idx_mat,       
    torch::Tensor idx_len,       
    torch::Tensor feat_idx,      
    torch::Tensor era_indices,  
    torch::Tensor active_classes, // <--- NEW argument
    int num_bins,         
    int K_tile_hint,
    int threads_per_block_hint
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_directional_split_kernel, "Best Split (CUDA)");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
    m.def("h_des_mc", &h_des_mc, "Multiclass per-node histogram with indirect class access");
}