#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm> // for std::min
#include <limits>

// =============================
// Common Helpers
// =============================

#define WARP_SIZE 32

// Helper for bfly kernel
static __device__ __forceinline__ void atomicAdd_f(float* addr, float val) {
  atomicAdd(addr, val);
}

// Helpers for smem kernel
static __forceinline__ __device__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
static __forceinline__ __device__ int warp_id() { return threadIdx.x >> 5; } // 5 = log2(32)

// Flattened indexer for GH/HH: [E, k, K, B] (contiguous)
static __forceinline__ __device__ size_t GH_idx(size_t e, size_t k, size_t K, size_t B,
                                                size_t e_id, size_t k_id, size_t c_id, size_t b_id) {
  // (((e_id * k + k_id) * K + c_id) * B + b_id)
  return (((e_id * k + k_id) * K + c_id) * B + b_id);
}

// CPU helper
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }


// ==========================================================
// KERNEL 1: Butterfly (Warp Intrinsics) - Robust Fallback
// ==========================================================
// This is your original, robust kernel. It's used when the
// histogram (E*B) is too large for shared memory.
//
// grid.x = k                 (per-feature blocks)
// grid.y = y_tiles           (striping over membership lists)
// grid.z = ceil_div(K, warps_per_block)
// blockDim.x = 32 * warps_per_block   (one warp == one class)
__global__ void h_des_mc_bfly_kernel(
    const int8_t* __restrict__ bin_indices, // [N,F]
    const float* __restrict__ grads,       // [N,K]
    const float* __restrict__ hess,        // [N,K]
    const int32_t* __restrict__ idx_mat,     // [K,Mmax]
    const int32_t* __restrict__ idx_len,     // [K]
    const int32_t* __restrict__ feat_idx,    // [k]
    const int32_t* __restrict__ era_idx,     // [N]
    float* __restrict__ GH,          // [E,k,K,B]
    float* __restrict__ HH,          // [E,k,K,B]
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
    const int stride_m        = WARP_SIZE * gridDim.y;
    int m = blockIdx.y * WARP_SIZE + lane;

    const unsigned full_mask = __activemask();

    for (; m < L; m += stride_m) {
        const bool valid = (m < L);
        const int jj = valid ? idx_mat[(size_t)c * (size_t)Mmax + (size_t)m] : -1;

        int   e = 0;
        int   b = 0;
        float g = 0.f;
        float h = 0.f;

        if (valid) {
            e = (int)era_idx[jj];
            const int8_t bin8 = bin_indices[(size_t)jj * (size_t)F + (size_t)f_global];
            b = (int)bin8;
            if (e < 0 || e >= E || b < 0 || b >= B) {
                g = 0.f; h = 0.f;
            } else {
                g = grads[(size_t)jj * (size_t)K + (size_t)c];
                h = hess [(size_t)jj * (size_t)K + (size_t)c];
            }
        }

        const uint32_t key = (valid && (e >= 0 && e < E) && (b >= 0 && b < B))
                           ? ((uint32_t)((uint32_t)e << 16) | (uint32_t)(uint32_t)b)
                           : 0xFFFFFFFFu;

        const unsigned group = __match_any_sync(full_mask, key);
        const int leader     = __ffs(group) - 1;

        // ===== FIX: Reverted to your original, correct float reduction =====
        float g_acc = g;
        float h_acc = h;
        // Note: use `group` as the active mask for shuffles
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
        // ===== End of FIX =====

        // Only the subgroup leader with a valid key updates global memory
        if (lane == leader && key != 0xFFFFFFFFu) {
            // Decode key from leader
            const int era  = (int)(key >> 16);
            const int bin  = (int)(key & 0xFFFFu);

            const size_t base = (((size_t)era * (size_t)k + (size_t)k_idx)
                               * (size_t)K + (size_t)c) * (size_t)B;
            atomicAdd_f(&GH[base + (size_t)bin], g_acc);
            atomicAdd_f(&HH[base + (size_t)bin], h_acc);
        }
    }
}


// ==========================================================
// KERNEL 2: Shared Memory - Fast Path
// ==========================================================
// This is your new, fast kernel. It's used when the histogram
// (E*B*K_tile) is small enough to fit in shared memory.
//
// Each block handles one (feature, class tile).
// - warps_per_block == tile_K (1 warp per class).
// - Shared memory layout: [tile_K, E, 2, B] floats (grad, count)
__global__ void _h_des_mc_smem(
    const int8_t* __restrict__ bin_idx,    // [N, F_master]
    const float* __restrict__ grads,      // [N, K_total]
    const float* __restrict__ hess,       // [N, K_total]
    const int32_t* __restrict__ idx_mat,    // [K_total, Mmax]
    const int32_t* __restrict__ idx_len,    // [K_total]
    const int32_t* __restrict__ feat_idx,   // [k]
    const int32_t* __restrict__ era_idx,    // [N]
    float* __restrict__ GH,         // [E, k, K_total, B]
    float* __restrict__ HH,         // [E, k, K_total, B]
    int N, int F_master,
    int K_total, int k, int B, int E,
    int Mmax, int tile_K
){
  const int k_local = blockIdx.x;                            // 0..k-1
  const int tile_id = blockIdx.y;                            // 0..ceil(K_total/tile_K)-1
  const int c0      = tile_id * tile_K;                      // starting class index in this tile
  const int c_rel   = warp_id();                             // 0..(tile_K-1) — warp-per-class
  const int lane    = lane_id();

  if (k_local >= k) return;

  const int K_here = min(tile_K, K_total - c0);              // classes actually in this tile
  if (c_rel >= K_here) return;                               // extra warps idle (last tile)

  // Global feature id
  const int f_global = feat_idx[k_local];

  extern __shared__ float shmem[];
  // Layout: [K_here, E, 2, B]
  const size_t per_class_elems = (size_t)E * 2 * (size_t)B;
  float* sh_base = shmem;

  // Zero shared memory collaboratively
  const int tpb = blockDim.x;
  const size_t total_elems = (size_t)K_here * per_class_elems;
  for (size_t i = threadIdx.x; i < total_elems; i += tpb) {
    sh_base[i] = 0.0f;
  }
  __syncthreads();

  // Each warp scans one class's sample list
  const int c_abs = c0 + c_rel;
  const int len   = idx_len[c_abs];
  const int32_t* row_list = &idx_mat[(size_t)c_abs * (size_t)Mmax];

  const size_t g_col_off = (size_t)c_abs;
  const size_t gh_col_stride = (size_t)K_total;
  float* sh_class = sh_base + (size_t)c_rel * per_class_elems;

  for (int p = lane; p < len; p += WARP_SIZE) {
    const int s = row_list[p];
    if ((unsigned)s >= (unsigned)N) continue;

    const int8_t b_raw = bin_idx[(size_t)s * (size_t)F_master + (size_t)f_global];
    if (b_raw < 0) continue;
    const int b = (int)b_raw;
    if (b >= B) continue;

    const int e = (int)era_idx[s];
    if ((unsigned)e >= (unsigned)E) continue;

    const float g = grads[(size_t)s * gh_col_stride + g_col_off];
    const float h = hess [(size_t)s * gh_col_stride + g_col_off];

    const size_t grad_ofs = ((size_t)e * 2 + 0) * (size_t)B + (size_t)b;
    const size_t cnt_ofs  = ((size_t)e * 2 + 1) * (size_t)B + (size_t)b;

    atomicAdd(&sh_class[grad_ofs], g);
    atomicAdd(&sh_class[cnt_ofs],  h);
  }
  __syncthreads(); // All warps must finish before flush

  // Flush shared to global (contention-free writes)
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
    const size_t out_idx = GH_idx((size_t)E, (size_t)k, (size_t)K_total, (size_t)B,
                                  e2, (size_t)k_local, (size_t)c_out, b2);
    if (ch == 0) {
      GH[out_idx] = val; // No atomic needed, we are the only writer
    } else {
      HH[out_idx] = val;
    }
  }
}


// ==========================================================
// LAUNCHER: Strategy Controller
// ==========================================================
// This function chooses which kernel to run based on histogram size
// and available shared memory.
std::vector<torch::Tensor> h_des_mc(
  torch::Tensor bin_indices,   // [N, F_master] int8
  torch::Tensor grads,         // [N, K] float32
  torch::Tensor hess,          // [N, K] float32
  torch::Tensor idx_mat,       // [K, Mmax] int32
  torch::Tensor idx_len,       // [K] int32
  torch::Tensor feat_idx,      // [k] int32
  torch::Tensor era_indices,   // [N] int32
  int num_bins,                // B
  int K_tile_hint,
  int threads_per_block_hint
){
  // --- 1. All input validation ---
  TORCH_CHECK(bin_indices.is_cuda() && grads.is_cuda() && hess.is_cuda() &&
              idx_mat.is_cuda() && idx_len.is_cuda() &&
              feat_idx.is_cuda() && era_indices.is_cuda(),
              "All tensors must be CUDA.");

  TORCH_CHECK(bin_indices.scalar_type() == torch::kInt8 || 
              bin_indices.scalar_type() == torch::kChar,
              "bin_indices must be int8/char.");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat,       "grads must be float32.");
  TORCH_CHECK(hess.scalar_type()  == torch::kFloat,       "hess must be float32.");
  TORCH_CHECK(idx_mat.scalar_type()== torch::kInt,        "idx_mat must be int32.");
  TORCH_CHECK(idx_len.scalar_type()== torch::kInt,        "idx_len must be int32.");
  TORCH_CHECK(feat_idx.scalar_type()== torch::kInt,       "feat_idx must be int32.");
  TORCH_CHECK(era_indices.scalar_type()== torch::kInt,    "era_indices must be int32.");
  TORCH_CHECK(bin_indices.dim()==2 && grads.dim()==2 && hess.dim()==2, "bin, grads, hess must be 2D.");
  TORCH_CHECK(idx_mat.dim()==2 && idx_len.dim()==1 && feat_idx.dim()==1 && era_indices.dim()==1,
              "idx_mat[K,Mmax], idx_len[K], feat_idx[k], era_indices[N].");

  const int N        = (int)bin_indices.size(0);
  const int F_master = (int)bin_indices.size(1);
  const int K_total  = (int)grads.size(1);
  const int K_total_h= (int)hess.size(1);
  TORCH_CHECK(K_total == K_total_h, "grads and hess must have same K columns.");
  TORCH_CHECK(idx_len.size(0) == K_total, "idx_len[K] must match K.");
  TORCH_CHECK(idx_mat.size(0) == K_total, "idx_mat[K,*] must match K.");
  const int Mmax     = (int)idx_mat.size(1);
  const int k        = (int)feat_idx.size(0);
  const int B        = num_bins;

  // Compute E = 1 + max(era_indices). This causes a D2H sync.
  const int E = (int)era_indices.max().item<int>() + 1;
  TORCH_CHECK(E >= 1, "Invalid era_indices (E<1).");

  // Allocate outputs: [E, k, K, B]
  auto opts = grads.options().dtype(torch::kFloat).memory_format(c10::MemoryFormat::Contiguous);
  torch::Tensor GH = torch::zeros({(int64_t)E, (int64_t)k, (int64_t)K_total, (int64_t)B}, opts);
  torch::Tensor HH = torch::zeros_like(GH);

  // --- 2. Device properties and strategy decision ---
  auto* prop = at::cuda::getCurrentDeviceProperties();
  const int maxThreadsPerBlock = prop->maxThreadsPerBlock;
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;

  const size_t bytes_per_class = (size_t)E * (size_t)B * 2 * sizeof(float);
  int tile_by_smem = (bytes_per_class == 0) ? 1 : (int)(smem_cap / bytes_per_class);
  
  bool use_shared = (tile_by_smem >= 1);

  auto stream = at::cuda::getCurrentCUDAStream();

  // --- 3. Launch the correct kernel ---
  if (use_shared) {
    //
    // ===== FAST PATH: Use Shared Memory Kernel (_h_des_mc_smem) =====
    //
    int tile_by_threads = maxThreadsPerBlock / WARP_SIZE;
    if (tile_by_threads < 1) tile_by_threads = 1;

    int K_tile = (K_tile_hint > 0) ? std::min({K_tile_hint, K_total, tile_by_threads})
                                   : std::min({8, K_total, tile_by_threads});
    
    K_tile = std::min(K_tile, tile_by_smem);
    if (K_tile < 1) K_tile = 1;

    int threads_per_block = (threads_per_block_hint > 0)
        ? threads_per_block_hint
        : (K_tile * WARP_SIZE);
    threads_per_block = std::min(threads_per_block, (maxThreadsPerBlock / WARP_SIZE) * WARP_SIZE);
    if (threads_per_block < WARP_SIZE) threads_per_block = WARP_SIZE;

    const int blocks_y = ceil_div_int(K_total, K_tile);
    dim3 grid((unsigned)k, (unsigned)blocks_y, 1u);
    dim3 block((unsigned)threads_per_block, 1u, 1u);
    
    const size_t smem_bytes = (size_t)K_tile * bytes_per_class;
    cudaFuncSetAttribute(_h_des_mc_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    _h_des_mc_smem<<<grid, block, smem_bytes, stream.stream()>>>(
      bin_indices.data_ptr<int8_t>(),
      grads.data_ptr<float>(),
      hess.data_ptr<float>(),
      idx_mat.data_ptr<int32_t>(),
      idx_len.data_ptr<int32_t>(),
      feat_idx.data_ptr<int32_t>(),
      era_indices.data_ptr<int32_t>(),
      GH.data_ptr<float>(),
      HH.data_ptr<float>(),
      N, F_master, K_total, k, B, E, Mmax, K_tile
    );
  
  } else {
    //
    // ===== ROBUST PATH: Use Butterfly Kernel (h_des_mc_bfly_kernel) =====
    //
    const int SM = prop->multiProcessorCount;
    int warps_per_block;
    if (threads_per_block_hint > 0) {
        warps_per_block = threads_per_block_hint / WARP_SIZE;
    } else if (K_tile_hint > 0) {
        warps_per_block = K_tile_hint;
    } else {
        warps_per_block = K_total >= 8 ? 8 : (K_total > 0 ? K_total : 1);
    }
    if (warps_per_block < 1) warps_per_block = 1;
    if (warps_per_block > 16) warps_per_block = 16;
    const int threads = warps_per_block * WARP_SIZE;

    const int tiles_z = ceil_div_int(K_total, warps_per_block);

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
    dim3 block((unsigned)threads, 1, 1);

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
        N, F_master, K_total, Mmax, k, E, B
    );
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "h_des_mc launch failed: ",
              cudaGetErrorString(cudaGetLastError()));

  return {GH, HH};
}