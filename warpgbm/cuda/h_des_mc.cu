#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

using torch::Tensor;

#define WARP 32
static __forceinline__ __device__ int lane_id() { return threadIdx.x & (WARP - 1); }
static __forceinline__ __device__ int warp_id() { return threadIdx.x >> 5; }

// Flattened indexer for GH/HH: [E, k, K, B] (contiguous)
static __forceinline__ __device__ size_t GH_idx(size_t e, size_t k, size_t K, size_t B,
                                                size_t e_id, size_t k_id, size_t c_id, size_t b_id) {
  // (((e_id * k + k_id) * K + c_id) * B + b_id)
  return (((e_id * k + k_id) * K + c_id) * B + b_id);
}

// =============================
// Shared-memory variant kernel
// =============================
// Each block handles one (feature, class tile).
// - warps_per_block == tile_K (1 warp per class).
// - Shared memory layout: [tile_K, E, 2, B] floats (grad, count)
__global__ void _h_des_mc_smem(
    const int8_t*  __restrict__ bin_idx,    // [N, F_master]
    const float*   __restrict__ grads,      // [N, K_total]
    const float*   __restrict__ hess,       // [N, K_total]
    const int32_t* __restrict__ idx_mat,    // [K_total, Mmax]
    const int32_t* __restrict__ idx_len,    // [K_total]
    const int32_t* __restrict__ feat_idx,   // [k]
    const int32_t* __restrict__ era_idx,    // [N]
    float*         __restrict__ GH,         // [E, k, K_total, B]
    float*         __restrict__ HH,         // [E, k, K_total, B]
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
  // We pack as a single contiguous array:
  //   class-major → era → ch(0=grad,1=count) → bin
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

  // Pointer offsets for grads/hess [N, K_total]
  const size_t g_col_off = (size_t)c_abs;       // column in [K_total]
  const size_t gh_col_stride = (size_t)K_total; // row-major stride

  // Shared base for this class
  float* sh_class = sh_base + (size_t)c_rel * per_class_elems;

  // Iterate over class samples with lane-stride
  for (int p = lane; p < len; p += WARP) {
    const int s = row_list[p];
    if ((unsigned)s >= (unsigned)N) continue;

    const int8_t b_raw = bin_idx[(size_t)s * (size_t)F_master + (size_t)f_global];
    if (b_raw < 0) continue;               // skip missing
    const int b = (int)b_raw;
    if (b >= B) continue;

    const int e = (int)era_idx[s];
    if ((unsigned)e >= (unsigned)E) continue;

    const float g = grads[(size_t)s * gh_col_stride + g_col_off];
    const float h = hess [(size_t)s * gh_col_stride + g_col_off];

    // sh index: [E, 2, B] → ((e*2 + ch)*B + b)
    const size_t grad_ofs = ((size_t)e * 2 + 0) * (size_t)B + (size_t)b;
    const size_t cnt_ofs  = ((size_t)e * 2 + 1) * (size_t)B + (size_t)b;

    // Many lanes may hit the same (e,b) → use shared atomics
    atomicAdd(&sh_class[grad_ofs], g);
    atomicAdd(&sh_class[cnt_ofs],  h);
  }
  __syncwarp();  // within-warp done; but we need block-wide sync before flush
  __syncthreads();

  // Flush shared to global (unique writer per element; no global atomics needed)
  // Each thread writes a subset of [K_here, E, 2, B]
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
      GH[out_idx] = GH[out_idx] + val;
    } else {
      HH[out_idx] = HH[out_idx] + val;
    }
  }
}

// =================================
// Global-atomic fallback kernel
// =================================
// Same mapping, but no shared memory. We atomically update GH/HH directly.
__global__ void _h_des_mc_global(
    const int8_t*  __restrict__ bin_idx,    // [N, F_master]
    const float*   __restrict__ grads,      // [N, K_total]
    const float*   __restrict__ hess,       // [N, K_total]
    const int32_t* __restrict__ idx_mat,    // [K_total, Mmax]
    const int32_t* __restrict__ idx_len,    // [K_total]
    const int32_t* __restrict__ feat_idx,   // [k]
    const int32_t* __restrict__ era_idx,    // [N]
    float*         __restrict__ GH,         // [E, k, K_total, B]
    float*         __restrict__ HH,         // [E, k, K_total, B]
    int N, int F_master,
    int K_total, int k, int B, int E,
    int Mmax, int tile_K
){
  const int k_local = blockIdx.x;
  const int tile_id = blockIdx.y;
  const int c0      = tile_id * tile_K;
  const int c_rel   = warp_id();
  const int lane    = lane_id();

  if (k_local >= k) return;

  const int K_here = min(tile_K, K_total - c0);
  if (c_rel >= K_here) return;

  const int f_global = feat_idx[k_local];

  const int c_abs = c0 + c_rel;
  const int len   = idx_len[c_abs];
  const int32_t* row_list = &idx_mat[(size_t)c_abs * (size_t)Mmax];

  const size_t g_col_off = (size_t)c_abs;
  const size_t gh_col_stride = (size_t)K_total;

  for (int p = lane; p < len; p += WARP) {
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

    const size_t out = GH_idx((size_t)E, (size_t)k, (size_t)K_total, (size_t)B,
                              (size_t)e, (size_t)k_local, (size_t)c_abs, (size_t)b);
    atomicAdd(&GH[out], g);
    atomicAdd(&HH[out], h);
  }
}

// ---------- helpers ----------
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// Host launcher choosing shared/global variant and autotuning K_tile/threads.
std::vector<Tensor> h_des_mc(
    Tensor bin_indices,   // [N, F_master] int8
    Tensor grads,         // [N, K] float32
    Tensor hess,          // [N, K] float32
    Tensor idx_mat,       // [K, Mmax] int32
    Tensor idx_len,       // [K] int32
    Tensor feat_idx,      // [k] int32
    Tensor era_indices,   // [N] int32
    int num_bins,         // B
    int K_tile_hint,
    int threads_per_block_hint
){
  TORCH_CHECK(bin_indices.is_cuda() && grads.is_cuda() && hess.is_cuda() &&
              idx_mat.is_cuda() && idx_len.is_cuda() &&
              feat_idx.is_cuda() && era_indices.is_cuda(),
              "All tensors must be CUDA.");

  TORCH_CHECK(bin_indices.scalar_type() == torch::kInt8,  "bin_indices must be int8.");
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

  // Compute E = 1 + max(era_indices)
  Tensor e_max = torch::amax(era_indices);
  int E;
  cudaMemcpy(&E, e_max.to(torch::kCPU).data_ptr<int>(), sizeof(int), cudaMemcpyHostToHost);
  E += 1;
  TORCH_CHECK(E >= 1, "Invalid era_indices (E<1).");

  // Allocate outputs: [E, k, K, B]
  auto opts = grads.options().dtype(torch::kFloat).memory_format(c10::MemoryFormat::Contiguous);
  Tensor GH = torch::zeros({(int64_t)E, (int64_t)k, (int64_t)K_total, (int64_t)B}, opts);
  Tensor HH = torch::zeros_like(GH);

  // Device properties
  auto* prop = at::cuda::getCurrentDeviceProperties();
  const int maxThreadsPerBlock = prop->maxThreadsPerBlock;
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;

  // Auto-tune tile size and threads
  const size_t bytes_per_class = (size_t)E * (size_t)B * 2 * sizeof(float);
  int tile_by_smem = (bytes_per_class == 0) ? 1 : (int)(smem_cap / bytes_per_class);
  if (tile_by_smem < 1) tile_by_smem = 0; // signals "use global" path

  int tile_by_threads = maxThreadsPerBlock / WARP; // 1 warp per class
  if (tile_by_threads < 1) tile_by_threads = 1;

  int K_tile = (K_tile_hint > 0) ? std::min({K_tile_hint, K_total, tile_by_threads})
                                 : std::min({8, K_total, tile_by_threads}); // 8 warps default

  bool use_shared = (tile_by_smem >= 1);
  if (use_shared) {
    K_tile = std::min(K_tile, tile_by_smem);
  } else {
    // shared mem can’t hold [E,B] per class → fall back to global atomics
    K_tile = std::min(K_tile, K_total);
  }
  if (K_tile < 1) K_tile = 1;

  int threads_per_block = (threads_per_block_hint > 0)
      ? threads_per_block_hint
      : (K_tile * WARP); // one warp per class
  // Cap by device limit & keep multiple of 32
  threads_per_block = std::min(threads_per_block, (maxThreadsPerBlock / WARP) * WARP);
  if (threads_per_block < WARP) threads_per_block = WARP;

  const int blocks_y = ceil_div_int(K_total, K_tile);
  dim3 grid((unsigned)k, (unsigned)blocks_y, 1u);
  dim3 block((unsigned)threads_per_block, 1u, 1u);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (use_shared) {
    const size_t smem_bytes = (size_t)K_tile * bytes_per_class;
    // Opt-in for dynamic shared memory on recent architectures
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
    // Global-atomic fallback, no dynamic shared memory
    _h_des_mc_global<<<grid, block, 0, stream.stream()>>>(
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
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "h_des_mc launch failed: ",
              cudaGetErrorString(cudaGetLastError()));

  return {GH, HH};
}
