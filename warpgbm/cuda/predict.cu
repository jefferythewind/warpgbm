#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bin_indices, // [N x F]
    const float *__restrict__ tree_tensor,  // [T x max_nodes x 6]
    int64_t N, int64_t F, int64_t T, int64_t max_nodes,
    float learning_rate,
    float *__restrict__ out // [N]
)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float sum = 0.0f;

    // Each thread handles one sample and iterates over ALL trees
    // This removes the need for atomicAdd to global memory
    for (int64_t t = 0; t < T; ++t)
    {
        const float *tree = tree_tensor + t * max_nodes * 6;

        int node_id = 0;
        while (true)
        {
            float is_leaf = tree[node_id * 6 + 4];
            if (is_leaf > 0.5f)
            {
                sum += tree[node_id * 6 + 5];
                break;
            }

            int feat = static_cast<int>(tree[node_id * 6 + 0]);
            int split_bin = static_cast<int>(tree[node_id * 6 + 1]);
            int left_id = static_cast<int>(tree[node_id * 6 + 2]);
            int right_id = static_cast<int>(tree[node_id * 6 + 3]);

            // Access bin index for this sample and feature
            // Since 'i' is constant for this thread, this is reasonably efficient
            int64_t bin_idx = i * F + feat;
            int8_t bin = bin_indices[bin_idx];

            node_id = (bin <= split_bin) ? left_id : right_id;
        }
    }

    // Single write to global memory
    // Use atomicAdd only if we are accumulating into an existing buffer (e.g. base_prediction)
    // But typical usage is out[i] initialized to base.
    atomicAdd(&out[i], learning_rate * sum);
}


void predict_with_forest(
    const at::Tensor &bin_indices,
    const at::Tensor &tree_tensor,
    float learning_rate,
    at::Tensor &out
)
{
    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t T = tree_tensor.size(0);
    int64_t max_nodes = tree_tensor.size(1);

    // Launch configuration based on N (samples), not N*T
    int threads_per_block = 256;
    int64_t blocks = (N + threads_per_block - 1) / threads_per_block;

    predict_forest_kernel<<<blocks, threads_per_block>>>(
        bin_indices.data_ptr<int8_t>(),
        tree_tensor.data_ptr<float>(),
        N, F, T, max_nodes,
        learning_rate,
        out.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA predict kernel failed: %s\n", cudaGetErrorString(err));
    }
}