#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void best_split_kernel_global_only(
    const float *__restrict__ G, // [F x B]
    const float *__restrict__ H, // [F x B]
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ best_gains, // [F]
    int *__restrict__ best_bins     // [F]
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F)
        return;

    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b)
    {
        G_total += G[f * B + b];
        H_total += H[f * B + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    float best_gain = min_split_gain;
    int best_bin = -1;

    for (int b = 0; b < B - 1; ++b)
    {
        G_L += G[f * B + b];
        H_L += H[f * B + b];
        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        if (H_L >= min_child_samples && H_R >= min_child_samples)
        {
            float gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps) - (G_total * G_total) / (H_total + eps);
            if (gain > best_gain)
            {
                best_gain = gain;
                best_bin = b;
            }
        }
    }

    best_gains[f] = best_gain;
    best_bins[f] = best_bin;
}

void launch_best_split_kernel_cuda(
    const at::Tensor &G, // [F x B]
    const at::Tensor &H, // [F x B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &best_gains, // [F], float32
    at::Tensor &best_bins,  // [F], int32
    int threads)
{
    int F = G.size(0);
    int B = G.size(1);

    int blocks = (F + threads - 1) / threads;

    best_split_kernel_global_only<<<blocks, threads>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        F,
        B,
        min_split_gain,
        min_child_samples,
        eps,
        best_gains.data_ptr<float>(),
        best_bins.data_ptr<int>());
}


// CUDA kernel: tie-breaker + era collapse + gain compute in one pass
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void tie_breaker_kernel(
    const float* __restrict__ G,      // [F * B * E]
    const float* __restrict__ H,      // [F * B * E]
    const int*   __restrict__ tie_feats, // [K]
    int F, int B, int E, int K,
    float min_gain, float min_child, float eps,
    float* __restrict__ out_gain,     // [K]
    int*   __restrict__ out_bin       // [K]
) {
    int idx = blockIdx.x;
    if (idx >= K) return;
    int f = tie_feats[idx];

    // per-feature collapse and gain scan
    extern __shared__ float s_mem[];  // size = 2 * B: first B for G_sum, next B for H_sum
    float* sG = s_mem;
    float* sH = s_mem + B;

    // collapse eras per-bin
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float sumG = 0.0f;
        float sumH = 0.0f;
        int base = (f * B + b) * E;
        for (int e = 0; e < E; ++e) {
            sumG += G[base + e];
            sumH += H[base + e];
        }
        sG[b] = sumG;
        sH[b] = sumH;
    }
    __syncthreads();

    // total sums
    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b) {
        G_total += sG[b];
        H_total += sH[b];
    }

    // scan bins to compute gain
    float G_L = 0.0f, H_L = 0.0f;
    float best_gain = min_gain;
    int best_bin = -1;
    for (int b = 0; b < B - 1; ++b) {
        G_L += sG[b];
        H_L += sH[b];
        float H_R = H_total - H_L;
        if (H_L >= min_child && H_R >= min_child) {
            float G_R = G_total - G_L;
            float gain = G_L * G_L / (H_L + eps)
                       + G_R * G_R / (H_R + eps)
                       - G_total * G_total / (H_total + eps);
            if (gain > best_gain) {
                best_gain = gain;
                best_bin = b;
            }
        }
    }

    out_gain[idx] = best_gain;
    out_bin[idx]  = best_bin;
}

// Launcher for the CUDA kernel
#include <ATen/ATen.h>
#include <torch/extension.h>

void launch_tie_breaker(
    const at::Tensor& G,    // [F, B, E]
    const at::Tensor& H,    // [F, B, E]
    const at::Tensor& tie_feats, // [K]
    float min_gain, float min_child, float eps,
    at::Tensor& out_gain,  // [K]
    at::Tensor& out_bin,   // [K]
    int threads_per_block
) {
    int F = G.size(0);
    int B = G.size(1);
    int E = G.size(2);
    int K = tie_feats.size(0);

    // flatten pointers
    const float* G_ptr = G.data_ptr<float>();
    const float* H_ptr = H.data_ptr<float>();
    const int*   T_ptr = tie_feats.data_ptr<int>();
    float* gain_ptr    = out_gain.data_ptr<float>();
    int*   bin_ptr     = out_bin.data_ptr<int>();

    int blocks = K;
    int threads = threads_per_block;
    int shared_bytes = 2 * B * sizeof(float);

    tie_breaker_kernel<<<blocks, threads, shared_bytes>>>(
        G_ptr, H_ptr, T_ptr,
        F, B, E, K,
        min_gain, min_child, eps,
        gain_ptr, bin_ptr
    );
    cudaDeviceSynchronize();
}