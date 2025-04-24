#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bin_indices, // [N x F]
    const float *__restrict__ tree_tensor,  // [T x max_nodes x 6]
    int N, int F, int T, int max_nodes,
    float learning_rate,
    float *__restrict__ out // [N]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_jobs = N * T;
    if (idx >= total_jobs)
        return;

    int i = idx % N; // sample index
    int t = idx / N; // tree index

    // if (i == 0 && t == 0)
    // {
    //     printf("[DEBUG] Thread (i=%d, t=%d): starting prediction\n", i, t);
    // }

    const float *tree = tree_tensor + t * max_nodes * 6;

    int node_id = 0;
    while (true)
    {
        float is_leaf = tree[node_id * 6 + 4];
        if (is_leaf > 0.5f)
        {
            float val = tree[node_id * 6 + 5];
            atomicAdd(&out[i], learning_rate * val);
            return;
        }
        int feat = static_cast<int>(tree[node_id * 6 + 0]);
        int split_bin = static_cast<int>(tree[node_id * 6 + 1]);
        int left_id = static_cast<int>(tree[node_id * 6 + 2]);
        int right_id = static_cast<int>(tree[node_id * 6 + 3]);

        int8_t bin = bin_indices[i * F + feat];
        node_id = (bin <= split_bin) ? left_id : right_id;
        // printf("sample %d, tree %d, feat %d, bin %d, split %d → %s\n", i, t, feat, bin, split_bin, (bin <= split_bin ? "L" : "R"));
    }
}

void predict_with_forest(
    const at::Tensor &bin_indices, // [N x F], int8
    const at::Tensor &tree_tensor, // [T x max_nodes x 6], float32
    float learning_rate,
    at::Tensor &out // [N], float32
)
{
    int N = bin_indices.size(0);
    int F = bin_indices.size(1);
    int T = tree_tensor.size(0);
    int max_nodes = tree_tensor.size(1);

    int total_jobs = N * T;
    int threads_per_block = 256;
    int blocks = (total_jobs + threads_per_block - 1) / threads_per_block;

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
