#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>

#define WARP_SIZE 32

// =============================
// Helpers
// =============================

template <typename T>
static __device__ __forceinline__ T load_ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

static __device__ __forceinline__ void atomicAdd_f(float* addr, float val) {
    atomicAdd(addr, val);
}

struct __align__(8) float2_packed {
    float x, y;
    __device__ __forceinline__ void operator+=(const float2_packed& other) {
        x += other.x;
        y += other.y;
    }
};

static __forceinline__ __device__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
static __forceinline__ __device__ int warp_id() { return threadIdx.x >> 5; }

static __forceinline__ __device__ size_t GH_idx(
    size_t e, size_t k, size_t K, size_t B,
    size_t e_id, size_t k_id, size_t c_id, size_t b_id
) {
    // layout: [E, k, K, B]
    return (((e_id * k + k_id) * K + c_id) * B + b_id);
}

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }


// ==========================================================
// KERNEL 1: Butterfly (global atomics, robust fallback)
// ==========================================================
__global__ void h_des_mc_bfly_kernel(
    const int8_t*  __restrict__ bin_indices,  // [N, F]
    const float*   __restrict__ grads,        // [N, K_stride]
    const float*   __restrict__ hess,         // [N, K_stride]
    const int32_t* __restrict__ idx_mat,      // [K_batch, Mmax]
    const int32_t* __restrict__ idx_len,      // [K_batch]
    const int32_t* __restrict__ feat_idx,     // [k]
    const int32_t* __restrict__ era_idx,      // [N]
    const int32_t* __restrict__ active_c,     // [K_batch] (local→global class map)
    float*         __restrict__ GH,           // [E, k, K_batch, B]
    float*         __restrict__ HH,           // [E, k, K_batch, B]
    const int N, const int F,
    const int K_batch, const int K_stride,
    const int Mmax, const int k, const int E, const int B
){
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_in_block   = threadIdx.x >> 5;
    const int lane            = threadIdx.x & (WARP_SIZE - 1);

    const int k_idx = blockIdx.x;  // feature slot
    if (k_idx >= k) return;

    const int f_global = feat_idx[k_idx];
    if (f_global < 0 || f_global >= F) return;

    const int class_tile = blockIdx.z;
    const int c_local    = class_tile * warps_per_block + warp_in_block;
    if (c_local >= K_batch) return;

    const int c_global = active_c[c_local];
    const int L        = idx_len[c_local];
    if (L <= 0) return;

    const int stride_m = WARP_SIZE * gridDim.y;
    int m = blockIdx.y * WARP_SIZE + lane;

    const unsigned full_mask = __activemask();

    for (; m < L; m += stride_m) {
        const int jj = idx_mat[(size_t)c_local * (size_t)Mmax + (size_t)m];

        int e = 0;
        int b = 0;
        float2_packed gh = {0.0f, 0.0f};
        bool valid = false;

        if ((unsigned)jj < (unsigned)N) {
            e = (int)era_idx[jj];
            if ((unsigned)e < (unsigned)E) {
                const int8_t bin8 = bin_indices[(size_t)jj * (size_t)F + (size_t)f_global];
                b = (int)bin8;
                if ((unsigned)b < (unsigned)B) {
                    const size_t g_idx = (size_t)jj * (size_t)K_stride + (size_t)c_global;
                    gh.x = load_ldg(&grads[g_idx]);
                    gh.y = load_ldg(&hess[g_idx]);
                    valid = true;
                }
            }
        }

        const uint32_t key = valid
            ? ((uint32_t)((uint32_t)e << 16) | (uint32_t)(uint32_t)b)
            : 0xFFFFFFFFu;

        const unsigned group = __match_any_sync(full_mask, key);

        // Subgroup butterfly reduction
        #pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            const float g_sh = __shfl_down_sync(group, gh.x, ofs);
            const float h_sh = __shfl_down_sync(group, gh.y, ofs);
            const int src_lane = lane + ofs;
            if (src_lane < WARP_SIZE && (group & (1u << src_lane))) {
                gh.x += g_sh;
                gh.y += h_sh;
            }
        }

        const int leader = __ffs(group) - 1;
        if (lane == leader && key != 0xFFFFFFFFu) {
            const int era_out = (int)(key >> 16);
            const int bin_out = (int)(key & 0xFFFFu);

            const size_t base = (((size_t)era_out * (size_t)k + (size_t)k_idx)
                               * (size_t)K_batch + (size_t)c_local) * (size_t)B;

            atomicAdd_f(&GH[base + (size_t)bin_out], gh.x);
            atomicAdd_f(&HH[base + (size_t)bin_out], gh.y);
        }
    }
}


// ==========================================================
// KERNEL 2: Shared-memory hist (fast path)
// - 1 warp per class in a tile
// - No warp-grouping; just atomics into shared memory
// ==========================================================
__global__ void _h_des_mc_smem(
    const int8_t*  __restrict__ bin_idx,     // [N, F_master]
    const float*   __restrict__ grads,       // [N, K_stride]
    const float*   __restrict__ hess,        // [N, K_stride]
    const int32_t* __restrict__ idx_mat,     // [K_batch, Mmax]
    const int32_t* __restrict__ idx_len,     // [K_batch]
    const int32_t* __restrict__ feat_idx,    // [k]
    const int32_t* __restrict__ era_idx,     // [N]
    const int32_t* __restrict__ active_c,    // [K_batch]
    float*         __restrict__ GH,          // [E, k, K_batch, B]
    float*         __restrict__ HH,          // [E, k, K_batch, B]
    int N, int F_master,
    int K_batch, int K_stride,
    int k, int B, int E,
    int Mmax, int tile_K
){
    const int k_local = blockIdx.x;
    if (k_local >= k) return;

    const int tile_id = blockIdx.y;
    const int c0      = tile_id * tile_K;

    const int warp_in_block = warp_id();
    const int lane          = lane_id();

    const int K_here = min(tile_K, K_batch - c0);
    if (warp_in_block >= K_here) return;

    const int f_global = feat_idx[k_local];
    if (f_global < 0 || f_global >= F_master) return;

    extern __shared__ float shmem[];
    const size_t per_class_elems = (size_t)E * 2 * (size_t)B;   // (grad,hess) per (era,bin)
    float* sh_base = shmem;

    // Zero shared memory for the active tile
    const int tpb = blockDim.x;
    const size_t total_elems = (size_t)K_here * per_class_elems;
    for (size_t i = threadIdx.x; i < total_elems; i += tpb) {
        sh_base[i] = 0.0f;
    }
    __syncthreads();

    const int c_local_abs = c0 + warp_in_block;
    const int c_global    = active_c[c_local_abs];
    const int len         = idx_len[c_local_abs];
    const int32_t* row_list = &idx_mat[(size_t)c_local_abs * (size_t)Mmax];

    float* sh_class = sh_base + (size_t)warp_in_block * per_class_elems;

    // Each warp scans its class's membership list
    for (int p = lane; p < len; p += WARP_SIZE) {
        const int s = row_list[p];
        if ((unsigned)s >= (unsigned)N) continue;

        const int e = (int)era_idx[s];
        if ((unsigned)e >= (unsigned)E) continue;

        const int8_t b_raw = bin_idx[(size_t)s * (size_t)F_master + (size_t)f_global];
        const int    b     = (int)b_raw;
        if ((unsigned)b >= (unsigned)B) continue;

        const size_t g_idx = (size_t)s * (size_t)K_stride + (size_t)c_global;
        const float g = load_ldg(&grads[g_idx]);
        const float h = load_ldg(&hess[g_idx]);

        const size_t grad_ofs = ((size_t)e * 2 + 0) * (size_t)B + (size_t)b;
        const size_t cnt_ofs  = ((size_t)e * 2 + 1) * (size_t)B + (size_t)b;

        atomicAdd(&sh_class[grad_ofs], g);
        atomicAdd(&sh_class[cnt_ofs],  h);
    }
    __syncthreads();

    // Flush shared histograms to global (no overlap between blocks in this path)
    for (size_t i = threadIdx.x; i < total_elems; i += tpb) {
        const size_t cls  = i / per_class_elems;
        const size_t rem1 = i % per_class_elems;
        const size_t e2   = rem1 / (2 * (size_t)B);
        const size_t rem2 = rem1 % (2 * (size_t)B);
        const size_t ch   = rem2 / (size_t)B;
        const size_t b2   = rem2 % (size_t)B;

        const float val = sh_base[i];
        if (val == 0.0f) continue;

        const int c_out = c0 + (int)cls;
        if (c_out >= K_batch) continue;

        const size_t out_idx = GH_idx(
            (size_t)E, (size_t)k, (size_t)K_batch, (size_t)B,
            e2, (size_t)k_local, (size_t)c_out, b2
        );

        if (ch == 0) {
            GH[out_idx] = val;
        } else {
            HH[out_idx] = val;
        }
    }
}


// ==========================================================
// Launcher
// ==========================================================
std::vector<torch::Tensor> h_des_mc(
    torch::Tensor bin_indices,    // [N, F_master] int8
    torch::Tensor grads,          // [N, K_stride] float32
    torch::Tensor hess,           // [N, K_stride] float32
    torch::Tensor idx_mat,        // [K_batch, Mmax] int32
    torch::Tensor idx_len,        // [K_batch] int32
    torch::Tensor feat_idx,       // [k] int32
    torch::Tensor era_indices,    // [N] int32
    torch::Tensor active_classes, // [K_batch] int32 (local→global)
    int num_bins,
    int K_tile_hint,
    int threads_per_block_hint
){
    TORCH_CHECK(
        bin_indices.is_cuda() && grads.is_cuda() && hess.is_cuda() &&
        idx_mat.is_cuda() && idx_len.is_cuda() &&
        feat_idx.is_cuda() && era_indices.is_cuda() &&
        active_classes.is_cuda(),
        "All tensors must be CUDA."
    );

    const int N        = (int)bin_indices.size(0);
    const int F_master = (int)bin_indices.size(1);
    const int K_stride = (int)grads.size(1);
    const int K_batch  = (int)idx_mat.size(0);

    TORCH_CHECK(idx_len.size(0) == K_batch,      "idx_len mismatch");
    TORCH_CHECK(active_classes.size(0) == K_batch, "active_classes mismatch");

    const int Mmax = (int)idx_mat.size(1);
    const int k    = (int)feat_idx.size(0);
    const int B    = num_bins;

    const int E = (int)era_indices.max().item<int>() + 1;
    TORCH_CHECK(E >= 1, "Invalid era_indices (E<1).");

    auto opts = grads.options()
        .dtype(torch::kFloat)
        .memory_format(c10::MemoryFormat::Contiguous);

    torch::Tensor GH = torch::zeros({(int64_t)E, (int64_t)k, (int64_t)K_batch, (int64_t)B}, opts);
    torch::Tensor HH = torch::zeros_like(GH);

    auto* prop = at::cuda::getCurrentDeviceProperties();
    const int maxThreadsPerBlock = prop->maxThreadsPerBlock;
    size_t smem_cap = prop->sharedMemPerBlockOptin
        ? (size_t)prop->sharedMemPerBlockOptin
        : (size_t)prop->sharedMemPerBlock;

    const size_t bytes_per_class = (size_t)E * (size_t)B * 2 * sizeof(float);
    int tile_by_smem = (bytes_per_class == 0)
        ? 1
        : (int)(smem_cap / bytes_per_class);

    const bool use_shared = (tile_by_smem >= 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (use_shared) {
        // ---------- FAST PATH: shared memory ----------
        int tile_by_threads = maxThreadsPerBlock / WARP_SIZE;
        if (tile_by_threads < 1) tile_by_threads = 1;

        int K_tile = (K_tile_hint > 0)
            ? std::min({K_tile_hint, K_batch, tile_by_threads})
            : std::min({8, K_batch, tile_by_threads});

        K_tile = std::min(K_tile, tile_by_smem);
        if (K_tile < 1) K_tile = 1;

        int threads_per_block = (threads_per_block_hint > 0)
            ? threads_per_block_hint
            : (K_tile * WARP_SIZE);

        threads_per_block =
            std::min(threads_per_block, (maxThreadsPerBlock / WARP_SIZE) * WARP_SIZE);
        if (threads_per_block < WARP_SIZE) threads_per_block = WARP_SIZE;

        const int blocks_y = ceil_div_int(K_batch, K_tile);

        dim3 grid((unsigned)k, (unsigned)blocks_y, 1u);
        dim3 block((unsigned)threads_per_block, 1u, 1u);

        const size_t smem_bytes = (size_t)K_tile * bytes_per_class;
        cudaFuncSetAttribute(
            _h_des_mc_smem,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes
        );

        _h_des_mc_smem<<<grid, block, smem_bytes, stream.stream()>>>(
            bin_indices.data_ptr<int8_t>(),
            grads.data_ptr<float>(),
            hess.data_ptr<float>(),
            idx_mat.data_ptr<int32_t>(),
            idx_len.data_ptr<int32_t>(),
            feat_idx.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            active_classes.data_ptr<int32_t>(),
            GH.data_ptr<float>(),
            HH.data_ptr<float>(),
            N, F_master, K_batch, K_stride,
            k, B, E, Mmax, K_tile
        );
    } else {
        // ---------- FALLBACK PATH: butterfly with global atomics ----------
        const int SM = prop->multiProcessorCount;

        int warps_per_block;
        if (threads_per_block_hint > 0) {
            warps_per_block = threads_per_block_hint / WARP_SIZE;
        } else if (K_tile_hint > 0) {
            warps_per_block = K_tile_hint;
        } else {
            warps_per_block = (K_batch >= 8) ? 8 : (K_batch > 0 ? K_batch : 1);
        }

        if (warps_per_block < 1)  warps_per_block = 1;
        if (warps_per_block > 16) warps_per_block = 16;

        const int threads = warps_per_block * WARP_SIZE;
        const int tiles_z = ceil_div_int(K_batch, warps_per_block);

        const int target_blocks_per_SM = 32;
        int min_total_blocks = SM * target_blocks_per_SM;
        int base_blocks = (k > 0 ? k : 1) * (tiles_z > 0 ? tiles_z : 1);
        int y_tiles = ceil_div_int(min_total_blocks, base_blocks);
        if (y_tiles < 1) y_tiles = 1;

        static constexpr int min_workload_per_thread = 128;
        int max_y_by_N = ceil_div_int(N, WARP_SIZE * min_workload_per_thread);
        if (max_y_by_N < 1) max_y_by_N = 1;
        if (y_tiles > max_y_by_N) y_tiles = max_y_by_N;
        if (y_tiles < 1) y_tiles = 1;

        dim3 grid((unsigned)k, (unsigned)y_tiles, (unsigned)tiles_z);
        dim3 block((unsigned)threads, 1u, 1u);

        h_des_mc_bfly_kernel<<<grid, block, 0, stream.stream()>>>(
            bin_indices.data_ptr<int8_t>(),
            grads.data_ptr<float>(),
            hess.data_ptr<float>(),
            idx_mat.data_ptr<int32_t>(),
            idx_len.data_ptr<int32_t>(),
            feat_idx.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            active_classes.data_ptr<int32_t>(),
            GH.data_ptr<float>(),
            HH.data_ptr<float>(),
            N, F_master, K_batch, K_stride,
            Mmax, k, E, B
        );
    }

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "h_des_mc launch failed: ",
                cudaGetErrorString(err));

    return {GH, HH};
}
