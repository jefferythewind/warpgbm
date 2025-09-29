#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// Warp-aggregated atomic: group threads in a warp that target the same key,
// sum their contributions in-register, and have only the leader do one atomicAdd.
// Works on Volta+ (match_any_sync). Falls back to per-thread if there's no grouping.
__device__ __forceinline__
void warp_agg_add(float* __restrict__ shG, float* __restrict__ shH,
                  int key, float g, float h)
{
    unsigned mask = __activemask();
    // If key is invalid, just return (no update).
    if (key < 0) return;

    // Find all lanes in the warp with the same 'key'
    unsigned group = __match_any_sync(mask, key);
    int lane = threadIdx.x & 31;
    int leader = __ffs(group) - 1; // first set bit

    // Sum g/h across the group (simple bit-walk)
    float gsum = g, hsum = h;
    unsigned remaining = group & ~(1u << lane);
    while (remaining) {
        int l = __ffs(remaining) - 1;
        gsum += __shfl_sync(mask, g, l);
        hsum += __shfl_sync(mask, h, l);
        remaining &= (remaining - 1);
    }

    // Only the leader lane does the atomic add into shared
    if (lane == leader) {
        atomicAdd(&shG[key], gsum);
        atomicAdd(&shH[key], hsum);
    }
}

// FM + warp-aggregated GH histogram (shared-memory block accumulation, then global flush).
// Shapes:
//   bin_fm:   [F_master, N]   int8 (feature-major bins)
//   grad_vec: [N]             float (reg: residual; binary: p - y)
//   hess_vec: [N]             float (reg: 1;         binary: p*(1-p))
//   sample_indices: [R]       int32 (rows in this node; arbitrary order ok)
//   feature_indices: [F_sub]  int32 (subset of features for this tree)
//   era_indices: [N]          int32 (0..E-1)
// Outputs:
//   grad_hist, hess_hist: [E, F_sub, B] float (raw counts/sums per bin; not prefix-scanned)
__global__ void histogram_gh_fm_warpagg_kernel(
    const int8_t*   __restrict__ bin_fm,        // [F_master, N]
    const float*    __restrict__ grad_vec,      // [N]
    const float*    __restrict__ hess_vec,      // [N]
    const int32_t*  __restrict__ sample_indices,// [R]
    const int32_t*  __restrict__ feature_indices,//[F_sub]
    const int32_t*  __restrict__ era_indices,   // [N]
    float*          __restrict__ grad_hist,     // [E, F_sub, B]
    float*          __restrict__ hess_hist,     // [E, F_sub, B]
    int64_t R, int64_t N, int64_t F_sub, int64_t B, int64_t E,
    int rows_per_thread)
{
    // One block handles (feature k, tile of rows)
    const int k    = blockIdx.x;                     // 0..F_sub-1
    const int feat = feature_indices[k];             // original feature id
    const int tid  = threadIdx.x;

    // Row tile this block works on
    const int rows_per_block = blockDim.x * rows_per_thread;
    const int row_start = (blockIdx.y * blockDim.x + tid) * rows_per_thread;

    // --- Shared memory: per-block histogram over (era, bin) ---
    // Layout: [E, B] for grad, then [E, B] for hess
    extern __shared__ float shmem[];
    float* shG = shmem;
    float* shH = shG + (E * B);

    // Zero shared hist (striped by threads)
    for (int i = tid; i < E * B; i += blockDim.x) {
        shG[i] = 0.f;
        shH[i] = 0.f;
    }
    __syncthreads();

    // --- Accumulate: each thread processes up to rows_per_thread samples in its tile ---
    #pragma unroll
    for (int u = 0; u < rows_per_thread; ++u) {
        int ridx = row_start + u;
        if (ridx >= R) break;

        const int sample = sample_indices[ridx];
        const int8_t bin = bin_fm[(int64_t)feat * N + sample]; // FM read
        const int e      = era_indices[sample];

        // Validate, compute flat key into [E, B]
        if ((unsigned)e >= (unsigned)E || (unsigned)bin >= (unsigned)B) continue;
        const int key = e * (int)B + (int)bin;

        const float g = grad_vec[sample];
        const float h = hess_vec[sample];

        // Warp-aggregated update into shared (reduces atomic pressure)
        warp_agg_add(shG, shH, key, g, h);
    }
    __syncthreads();

    // --- Flush shared histogram to global with one atomic per (e,bin) per block ---
    for (int i = tid; i < E * B; i += blockDim.x) {
        const int e   = i / B;
        const int bin = i % B;
        const int64_t out_idx = ((int64_t)e * F_sub + k) * B + bin;
        atomicAdd(&grad_hist[out_idx], shG[i]);
        atomicAdd(&hess_hist[out_idx], shH[i]);
    }
}

// Launcher
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
    int rows_per_thread = 4)
{
    TORCH_CHECK(bin_fm.is_cuda() && bin_fm.dtype()==at::kChar && bin_fm.is_contiguous(), "bin_fm [F,N] int8 contiguous");
    TORCH_CHECK(grad_vec.is_cuda() && grad_vec.dtype()==at::kFloat && grad_vec.is_contiguous(), "grad_vec float contiguous");
    TORCH_CHECK(hess_vec.is_cuda() && hess_vec.dtype()==at::kFloat && hess_vec.is_contiguous(), "hess_vec float contiguous");
    TORCH_CHECK(sample_indices.is_cuda() && sample_indices.dtype()==at::kInt && sample_indices.is_contiguous(), "sample_indices int32 contiguous");
    TORCH_CHECK(feature_indices.is_cuda() && feature_indices.dtype()==at::kInt && feature_indices.is_contiguous(), "feature_indices int32 contiguous");
    TORCH_CHECK(era_indices.is_cuda() && era_indices.dtype()==at::kInt && era_indices.is_contiguous(), "era_indices int32 contiguous");
    TORCH_CHECK(grad_hist.is_cuda() && hess_hist.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK((threads_per_block % 32) == 0, "threads_per_block must be a multiple of 32");

    const int64_t R     = sample_indices.size(0);
    const int64_t N     = bin_fm.size(1);
    const int64_t F_sub = feature_indices.size(0);
    const int64_t E     = grad_hist.size(0);
    const int      B     = num_bins;

    const int64_t rows_per_blk = (int64_t)threads_per_block * rows_per_thread;
    const int64_t tiles = (R + rows_per_blk - 1) / rows_per_blk;

    dim3 grid((unsigned)F_sub, (unsigned)tiles, 1);
    dim3 block((unsigned)threads_per_block, 1, 1);

    // Shared mem: two planes [E,B] (grad + hess)
    const size_t shmem_bytes = (size_t)(2 * E * B) * sizeof(float);

    histogram_gh_fm_warpagg_kernel<<<grid, block, shmem_bytes>>>(
        bin_fm.data_ptr<int8_t>(),
        grad_vec.data_ptr<float>(),
        hess_vec.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        R, N, F_sub, B, E,
        rows_per_thread
    );

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA histogram_gh_fm_warpagg launch failed: %s\n", cudaGetErrorString(err));
    }
}
