#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void best_split_kernel_levelwise(
    const float *__restrict__ G, // [N x F x B] flattened
    const float *__restrict__ H,
    const int *__restrict__ node_ids, // [N], maps kernel-local node index -> global node index
    int N, int F, int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ split_gains, // [global_nodes x F]
    int *__restrict__ split_bins     // [global_nodes x F]
)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= N * F)
        return;

    int local_n = global_id / F;
    int f = global_id % F;
    int n = node_ids[local_n]; // actual global node id

    int base_idx = n * F * B + f * B; // indexing into local G/H

    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b)
    {
        G_total += G[base_idx + b];
        H_total += H[base_idx + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    float best_gain = min_split_gain;
    int best_bin = -1;

    for (int b = 0; b < B - 1; ++b)
    {
        G_L += G[base_idx + b];
        H_L += H[base_idx + b];
        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        if (H_L >= min_child_samples && H_R >= min_child_samples)
        {
            float gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps);
            if (gain > best_gain)
            {
                best_gain = gain;
                best_bin = b;
            }
        }
    }

    int out_idx = n * F + f;
    split_gains[out_idx] = best_gain;
    split_bins[out_idx] = best_bin;
}

void find_splits_for_level(
    const at::Tensor &node_ids, // [N] global node ids
    const at::Tensor &G,        // [N x F x B]
    const at::Tensor &H,        // [N x F x B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &split_gains, // [max_nodes x F]
    at::Tensor &split_bins,  // [max_nodes x F]
    int threads_per_block = 256)
{
    int N = node_ids.size(0); // FIXED: use node_ids length, not G.size(0)
    int F = G.size(1);
    int B = G.size(2);

    int total_jobs = N * F;
    int blocks = (total_jobs + threads_per_block - 1) / threads_per_block;

    dim3 grid(blocks);
    dim3 threads(threads_per_block);

    best_split_kernel_levelwise<<<grid, threads>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        node_ids.data_ptr<int>(),
        N, F, B,
        min_split_gain,
        min_child_samples,
        eps,
        split_gains.data_ptr<float>(),
        split_bins.data_ptr<int>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA split kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}