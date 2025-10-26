// h_des_mc.cu — multiclass histogram (E,k,K,B) with warp-aggregated butterfly updates.
// Always-on butterfly route (no toggles).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

#define WARP_SIZE 32

// ---------- small helpers ----------
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// Atomic add for float is built-in; keep alias for clarity
static __device__ __forceinline__ void atomicAdd_f(float* addr, float val) {
  atomicAdd(addr, val);
}

// ---------- Kernel ----------
// bin_indices: [N, F] int8
// grads, hess: [N, K] float32
// idx_mat:     [K, Mmax] int32  (per-class membership indices)
// idx_len:     [K] int32         (valid length per class)
// feat_idx:    [k] int32         (feature ids)
// era_idx:     [N] int32         (0..E-1)
// GH, HH:      [E, k, K, B] float32  (output)
//
// grid.x = k                 (per-feature blocks)
// grid.y = y_tiles           (striping over membership lists)
// grid.z = ceil_div(K, warps_per_block)
// blockDim.x = 32 * warps_per_block   (one warp == one class)
__global__ void h_des_mc_bfly_kernel(
    const int8_t*  __restrict__ bin_indices, // [N,F]
    const float*   __restrict__ grads,       // [N,K]
    const float*   __restrict__ hess,        // [N,K]
    const int32_t* __restrict__ idx_mat,     // [K,Mmax]
    const int32_t* __restrict__ idx_len,     // [K]
    const int32_t* __restrict__ feat_idx,    // [k]
    const int32_t* __restrict__ era_idx,     // [N]
    float*         __restrict__ GH,          // [E,k,K,B]
    float*         __restrict__ HH,          // [E,k,K,B]
    const int N, const int F, const int K,
    const int Mmax, const int k, const int E, const int B
){
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_in_block   = threadIdx.x >> 5;     // 0..warps_per_block-1
    const int lane            = threadIdx.x & 31;     // 0..31

    const int k_idx           = blockIdx.x;           // feature slot
    if (k_idx >= k) return;
    const int f_global        = feat_idx[k_idx];
    if (f_global < 0 || f_global >= F) return;

    const int class_tile      = blockIdx.z;
    const int c               = class_tile * warps_per_block + warp_in_block; // class id
    if (c >= K) return;

    const int L               = idx_len[c];
    if (L <= 0) return;

    // Process membership list in stripes split by grid.y.
    // Each warp lane grabs position m = (blockIdx.y * 32 + lane) + t * 32 * gridDim.y
    const int stride_m        = WARP_SIZE * gridDim.y;
    int m = blockIdx.y * WARP_SIZE + lane;

    // Fast path constants
    const unsigned full_mask = __activemask();

    for (; m < L; m += stride_m) {
        // Valid lane?
        const bool valid = (m < L);
        // Load sample id
        const int jj = valid ? idx_mat[(size_t)c * (size_t)Mmax + (size_t)m] : -1;

        // Load data for this sample
        int   e = 0;
        int   b = 0;
        float g = 0.f;
        float h = 0.f;

        if (valid) {
            // era and bin; guard bin range
            e = (int)era_idx[jj];
            const int8_t bin8 = bin_indices[(size_t)jj * (size_t)F + (size_t)f_global];
            b = (int)bin8;
            if (e < 0 || e >= E || b < 0 || b >= B) {
                // mark invalid so it won't contribute
                // (we'll still participate in shuffles safely)
                g = 0.f; h = 0.f;
            } else {
                g = grads[(size_t)jj * (size_t)K + (size_t)c];
                h = hess [(size_t)jj * (size_t)K + (size_t)c];
            }
        }

        // Build 32-bit key for match_any: (era << 16) | bin   (require E <= 65535, B <= 65535)
        // Invalid lanes use 0xFFFFFFFF sentinel (won't alias with valid since b < 65535).
        const uint32_t key = (valid && (e >= 0 && e < E) && (b >= 0 && b < B))
                           ? ((uint32_t)((uint32_t)e << 16) | (uint32_t)(uint32_t)b)
                           : 0xFFFFFFFFu;

        // Group lanes in warp that share the same (e,b)
        const unsigned group = __match_any_sync(full_mask, key);
        const int leader     = __ffs(group) - 1;

        // Butterfly-style reduction inside the group mask
        float g_acc = g;
        float h_acc = h;
        // Note: use `group` as the active mask for shuffles to restrict to our subgroup
        #pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            const float g_sh = __shfl_down_sync(group, g_acc, ofs);
            const float h_sh = __shfl_down_sync(group, h_acc, ofs);
            // Only lanes with a partner in the subgroup accumulate
            if (((lane + ofs) < 32) && (group & (1u << (lane + ofs)))) {
                g_acc += g_sh;
                h_acc += h_sh;
            }
        }

        // Only the subgroup leader with a valid key updates global memory
        if (lane == leader && key != 0xFFFFFFFFu) {
            // Decode key
            const int era  = (int)(key >> 16);
            const int bin  = (int)(key & 0xFFFFu);

            // Flat index for GH/HH [E,k,K,B], row-major:
            // idx = (((e * k + k_idx) * K + c) * B + b)
            const size_t base = (((size_t)era * (size_t)k + (size_t)k_idx)
                               *  (size_t)K + (size_t)c) * (size_t)B;
            atomicAdd_f(&GH[base + (size_t)bin], g_acc);
            atomicAdd_f(&HH[base + (size_t)bin], h_acc);
        }
    }
}

// ---------- Launcher ----------
std::vector<torch::Tensor> h_des_mc(
    torch::Tensor bin_indices,     // [N,F] int8, CUDA
    torch::Tensor grads,           // [N,K] float32, CUDA
    torch::Tensor hess,            // [N,K] float32, CUDA
    torch::Tensor idx_mat,         // [K,Mmax] int32, CUDA
    torch::Tensor idx_len,         // [K] int32, CUDA
    torch::Tensor feat_idx,        // [k] int32, CUDA
    torch::Tensor era_indices,     // [N] int32, CUDA
    int num_bins,                  // B
    int K_tile_hint,               // optional, can be 0
    int threads_per_block_hint     // optional, can be 0
){
    TORCH_CHECK(bin_indices.is_cuda() && grads.is_cuda() && hess.is_cuda() &&
                idx_mat.is_cuda() && idx_len.is_cuda() &&
                feat_idx.is_cuda() && era_indices.is_cuda(),
                "All tensors must be CUDA.");

    TORCH_CHECK(bin_indices.dim()==2, "bin_indices must be [N,F].");
    TORCH_CHECK(grads.dim()==2 && hess.dim()==2, "grads/hess must be [N,K].");
    TORCH_CHECK(idx_mat.dim()==2, "idx_mat must be [K,Mmax].");
    TORCH_CHECK(idx_len.dim()==1, "idx_len must be [K].");
    TORCH_CHECK(feat_idx.dim()==1, "feat_idx must be [k].");
    TORCH_CHECK(era_indices.dim()==1, "era_indices must be [N].");

    TORCH_CHECK(bin_indices.scalar_type() == c10::ScalarType::Char ||
                bin_indices.scalar_type() == c10::ScalarType::Byte,
                "bin_indices must be int8/uint8.");
    TORCH_CHECK(grads.scalar_type() == c10::ScalarType::Float &&
                hess.scalar_type()  == c10::ScalarType::Float,
                "grads/hess must be float32.");
    TORCH_CHECK(idx_mat.scalar_type()  == c10::ScalarType::Int &&
                idx_len.scalar_type()  == c10::ScalarType::Int &&
                feat_idx.scalar_type() == c10::ScalarType::Int &&
                era_indices.scalar_type() == c10::ScalarType::Int,
                "idx_mat/idx_len/feat_idx/era_indices must be int32.");

    const int N    = (int)bin_indices.size(0);
    const int F    = (int)bin_indices.size(1);
    const int K    = (int)grads.size(1);
    const int K2   = (int)hess.size(1);
    TORCH_CHECK((int)grads.size(0) == N && (int)hess.size(0) == N &&
                K2 == K, "grads/hess shapes must match [N,K].");

    TORCH_CHECK((int)idx_mat.size(0) == K, "idx_mat.size(0) must equal K.");
    const int Mmax = (int)idx_mat.size(1);
    TORCH_CHECK((int)idx_len.size(0) == K, "idx_len.size(0) must equal K.");

    const int k     = (int)feat_idx.size(0);
    const int E     = (int)era_indices.max().item<int>() + 1;
    const int B     = num_bins;
    TORCH_CHECK(B > 0 && B <= 127, "num_bins must be in (0, 127].");
    TORCH_CHECK(E > 0, "E (num eras) must be > 0.");
    TORCH_CHECK(E <= 65535, "butterfly key packs era into 16 bits; require E <= 65535.");

    // Allocate outputs [E, k, K, B] float32
    auto opts = grads.options().dtype(torch::kFloat).memory_format(c10::MemoryFormat::Contiguous);
    auto GH = torch::zeros({(long long)E, (long long)k, (long long)K, (long long)B}, opts);
    auto HH = torch::zeros({(long long)E, (long long)k, (long long)K, (long long)B}, opts);

    // --- choose launch dims ---
    auto* prop = at::cuda::getCurrentDeviceProperties();
    const int SM = prop->multiProcessorCount;

    // warps per block (one warp per class in the tile)
    int warps_per_block;
    if (threads_per_block_hint > 0) {
        TORCH_CHECK(threads_per_block_hint % WARP_SIZE == 0, "threads_per_block_hint must be multiple of 32.");
        warps_per_block = threads_per_block_hint / WARP_SIZE;
    } else if (K_tile_hint > 0) {
        warps_per_block = K_tile_hint;
    } else {
        // heuristic: up to 8 warps per block, but not more than K
        warps_per_block = K >= 8 ? 8 : (K > 0 ? K : 1);
    }
    if (warps_per_block < 1) warps_per_block = 1;
    if (warps_per_block > 16) warps_per_block = 16; // safety cap
    const int threads = warps_per_block * WARP_SIZE;

    // grid.z tiles over classes
    const int tiles_z = ceil_div_int(K, warps_per_block);

    // Aim for ~32 blocks per SM; spread remaining parallelism over grid.y
    const int target_blocks_per_SM = 32;
    int min_total_blocks = SM * target_blocks_per_SM;
    int base_blocks = (k > 0 ? k : 1) * (tiles_z > 0 ? tiles_z : 1);
    int y_tiles = ceil_div_int(min_total_blocks, base_blocks);
    if (y_tiles < 1) y_tiles = 1;

    // Avoid oversplitting when nodes are tiny: bound by N (using coarse heuristic)
    static constexpr int min_workload_per_thread = 128; // similar to h0_des_butterfly
    int max_y_by_N = ceil_div_int(N, WARP_SIZE * min_workload_per_thread);
    if (max_y_by_N < 1) max_y_by_N = 1;
    if (y_tiles > max_y_by_N) y_tiles = max_y_by_N;
    if (y_tiles < 1) y_tiles = 1;

    dim3 grid((unsigned)k, (unsigned)y_tiles, (unsigned)tiles_z);
    dim3 block((unsigned)threads, 1, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch
    h_des_mc_bfly_kernel<<<grid, block, 0, stream.stream()>>>(
        bin_indices.data_ptr<int8_t>(),
        grads.data_ptr<float>(),
        hess.data_ptr<float>(),
        idx_mat.data_ptr<int32_t>(),
        idx_len.data_ptr<int32_t>(),
        feat_idx.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        GH.data_ptr<float>(),
        HH.data_ptr<float>(),
        N, F, K, Mmax, k, E, B
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "h_des_mc kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return {GH, HH};
}
