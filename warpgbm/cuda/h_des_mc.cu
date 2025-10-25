// warpgbm/cuda/h_des_mc.cu
// Multiclass histogram (era-compacted + class-batched) with warp-level butterfly aggregation.
// - One-pass compactor (_et_compact_bfly_onepass): groups by era using match_any + atomicAdd reservation
// - Histogram kernel (_h_des_mc_bfly): warp=class, per-warp shared rows, warp-aggregated updates by bin
// - Root fast-path: if eras are sorted, skip compaction and fabricate contiguous per-era ranges
//
// Returns stacked tensor [2, E, k, K, B] (grad, hess)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------- Helpers ----------------
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

static inline int choose_warps_per_block(size_t smem_cap_bytes, int B, int pad = 1) {
  // per-warp shared: sG + sH = 2 * (B + pad) floats
  size_t per_warp = 2ull * (size_t)(B + pad) * sizeof(float);
  int wpb = (int)(smem_cap_bytes / per_warp);
  if (wpb < 1) wpb = 1;
  if (wpb > 16) wpb = 16;
  // keep blockDim sane and balanced
  if (wpb >= 12) return 12;
  if (wpb >= 8)  return 8;
  if (wpb >= 4)  return 4;
  return 2;
}

// ==========================
// 1) One-pass era compaction (butterfly)
//    heads[c,e] is zeroed before launch; becomes the final count after launch.
// ==========================
__global__ void _et_compact_bfly_onepass(
    const int32_t* __restrict__ idx_mat,    // [K, Mmax]
    const int32_t* __restrict__ idx_len,    // [K]
    const int32_t* __restrict__ era_of_row, // [N]
    int K, int Mmax, int E,
    int32_t* __restrict__ heads,            // [K, E], zeroed; AFTER: final counts
    int32_t* __restrict__ idx_out,          // [K, Mmax]
    int rows_per_thread)                    // e.g., 2 or 4
{
  const int c = blockIdx.x;                  // class id
  if (c >= K) return;
  const int len = idx_len[c];
  if (len <= 0) return;

  const int warp  = threadIdx.x >> 5;        // 0..W-1
  const int lane  = threadIdx.x & 31;        // 0..31
  const int warps = blockDim.x >> 5;

  const int rows_per_block = blockDim.x * rows_per_thread;
  const int tile = blockIdx.y;
  const int start = tile * rows_per_block;
  if (start >= len) return;
  const int end = min(start + rows_per_block, len);

  // strip-mine this tile across warps; each lane gets distinct i
  for (int i0 = start + warp * 32; i0 < end; i0 += warps * 32) {
    #pragma unroll
    for (int step = 0; step < rows_per_thread; ++step) {
      int i = i0 + lane + step * blockDim.x;
      if (i >= end) break;

      const int n = idx_mat[(size_t)c * (size_t)Mmax + i];
      const int e = era_of_row[n];                 // grouping key

      // group lanes with same era inside this warp
      unsigned full   = __activemask();
      unsigned peers  = __match_any_sync(full, e);
      const int leader= __ffs(peers) - 1;
      const int gsize = __popc(peers);

      // leader reserves contiguous block for the group in heads[c,e]
      int base = 0;
      if (lane == leader) {
        base = atomicAdd(&heads[(size_t)c * (size_t)E + e], gsize);
      }
      // broadcast base to the group only
      base = __shfl_sync(peers, base, leader);

      // rank within group (prefix count)
      const unsigned lower = peers & ((1u << lane) - 1u);
      const int rank = __popc(lower);

      // scatter into class-compacted output
      const int pos = base + rank;
      idx_out[(size_t)c * (size_t)Mmax + pos] = n;
    }
  }
}

// Heads (counts) -> per-class era offsets (exclusive scan)
__global__ void _scan_counts_to_offsets(
    const int32_t* __restrict__ counts, // [K,E]
    int K, int E,
    int32_t* __restrict__ off_era)      // [K,E+1]
{
  const int c = blockIdx.x;
  if (c >= K) return;
  int acc = 0;
  for (int e = 0; e < E; ++e) {
    off_era[(size_t)c * (size_t)(E + 1) + e] = acc;
    acc += counts[(size_t)c * (size_t)E + e];
  }
  off_era[(size_t)c * (size_t)(E + 1) + E] = acc;
}

// ==========================
// 2) Class-batched histogram (butterfly), warp=class
// ==========================
__global__ void _h_des_mc_bfly(
    const int8_t*  __restrict__ bin_idx,   // [N,F]
    const float*   __restrict__ G,         // [N,K]
    const float*   __restrict__ H,         // [N,K]
    const int32_t* __restrict__ feat_idx,  // [k]
    const int32_t* __restrict__ idx_out,   // [K,Mmax]
    const int32_t* __restrict__ off_era,   // [K,E+1]
    float* __restrict__ GH,                // [E,k,K,B]
    float* __restrict__ HH,                // [E,k,K,B]
    int N, int F, int K, int E, int k, int B, int Mmax,
    int pad)                               // shared-row padding (avoid bank conflicts)
{
  const int j    = blockIdx.x;                // feature in subset
  const int era  = blockIdx.y;                // era id
  const int wpb  = blockDim.x >> 5;           // warps per block
  const int warp = threadIdx.x >> 5;          // 0..wpb-1
  const int lane = threadIdx.x & 31;          // 0..31
  const int c    = blockIdx.z * wpb + warp;   // class id
  if (j >= k || era >= E || c >= K) return;

  extern __shared__ float sm[];
  const int stride = (B + pad);
  float* sG = sm;                              // [wpb * (B+pad)]
  float* sH = sm + (size_t)wpb * stride;       // [wpb * (B+pad)]

  // zero this warp's row
  for (int b = lane; b < B; b += 32) {
    sG[warp * stride + b] = 0.f;
    sH[warp * stride + b] = 0.f;
  }
  __syncthreads();

  const int f = feat_idx[j];
  const int32_t* off = off_era + (size_t)c * (size_t)(E + 1);
  const int start = off[era];
  const int end   = off[era + 1];

  // stride the contiguous era segment, lane-strided
  for (int m = start + lane; m < end; m += 32) {
    const int n   = idx_out[(size_t)c * (size_t)Mmax + m];
    const int bin = (int)(uint8_t)bin_idx[(size_t)n * (size_t)F + f];
    const float g = G[(size_t)n * (size_t)K + c];
    const float h = H[(size_t)n * (size_t)K + c];

    // warp-aggregated updates by 'bin' (era is fixed in this block)
    unsigned full   = __activemask();
    unsigned peers  = __match_any_sync(full, bin);
    const int leader= __ffs(peers) - 1;
    const float g_sum = __reduce_add_sync(peers, g);
    const float h_sum = __reduce_add_sync(peers, h);

    if (lane == leader) {
      atomicAdd(&sG[warp * stride + bin], g_sum);
      atomicAdd(&sH[warp * stride + bin], h_sum);
    }
  }
  __syncthreads();

  // Flush to global [E,k,K,B] (each warp writes disjoint class slice)
  const size_t base = ((((size_t)era * (size_t)k) + (size_t)j) * (size_t)K + (size_t)c) * (size_t)B;
  for (int b = lane; b < B; b += 32) {
    GH[base + b] = sG[warp * stride + b];
    HH[base + b] = sH[warp * stride + b];
  }
}

// ==========================
// 3) Public orchestrator
// ==========================
// API:
//   h_des_mc(bin_idx[N,F]int8, era_of_row[N]int32, G[N,K]f32, H[N,K]f32,
//            feat_idx[k]int32, idx_mat[K,Mmax]int32, idx_len[K]int32,
//            era_ends[E]int32 (exclusive ends for root fast-path),
//            B, enable_compact, root_fastpath) -> [2,E,k,K,B]

torch::Tensor h_des_mc(
    torch::Tensor bin_idx,
    torch::Tensor era_of_row,
    torch::Tensor G,
    torch::Tensor H,
    torch::Tensor feat_idx,
    torch::Tensor idx_mat,
    torch::Tensor idx_len,
    torch::Tensor era_ends,
    int B,
    bool enable_compact,
    bool root_fastpath)
{
  TORCH_CHECK(bin_idx.is_cuda() && era_of_row.is_cuda() && G.is_cuda() && H.is_cuda()
           && feat_idx.is_cuda() && idx_mat.is_cuda() && idx_len.is_cuda(),
           "All tensors must be CUDA.");
  TORCH_CHECK(bin_idx.scalar_type() == torch::kInt8,    "bin_idx must be int8.");
  TORCH_CHECK(era_of_row.scalar_type() == torch::kInt32,"era_of_row must be int32.");
  TORCH_CHECK(G.scalar_type() == torch::kFloat && H.scalar_type() == torch::kFloat,
              "G and H must be float32.");
  TORCH_CHECK(feat_idx.scalar_type() == torch::kInt32,  "feat_idx must be int32.");
  TORCH_CHECK(idx_mat.scalar_type() == torch::kInt32 && idx_len.scalar_type() == torch::kInt32,
              "idx_mat/idx_len must be int32.");
  TORCH_CHECK(era_ends.scalar_type() == torch::kInt32,  "era_ends must be int32.");

  TORCH_CHECK(bin_idx.dim()==2, "bin_idx [N,F]");
  TORCH_CHECK(G.dim()==2 && H.dim()==2 && G.sizes()==H.sizes(), "G/H [N,K] and same shape");
  TORCH_CHECK(feat_idx.dim()==1, "feat_idx [k]");
  TORCH_CHECK(idx_mat.dim()==2 && idx_len.dim()==1, "idx_mat [K,Mmax], idx_len [K]");
  TORCH_CHECK(era_ends.dim()==1, "era_ends [E]");

  const int N = (int)bin_idx.size(0);
  const int F = (int)bin_idx.size(1);
  const int K = (int)G.size(1);
  const int k = (int)feat_idx.size(0);
  const int Mmax = (int)idx_mat.size(1);
  const int E = (int)era_ends.size(0);

  TORCH_CHECK(G.size(0) == N && H.size(0) == N, "G/H N mismatch with bin_idx");
  TORCH_CHECK(idx_mat.size(0) == K && idx_len.size(0) == K, "idx_* K mismatch");

  // Output tensors
  auto GH = torch::zeros({ (long long)E, (long long)k, (long long)K, (long long)B }, G.options());
  auto HH = torch::zeros_like(GH);

  // Shared-memory sizing & block shape for histogram
  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;
  const int PAD = 1; // bank-conflict padding per row
  const int wpb = choose_warps_per_block(smem_cap, B, PAD);
  const size_t smem_hist = 2ull * (size_t)wpb * (size_t)(B + PAD) * sizeof(float);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Heuristic: compact when many rows and multiple eras
  const int total_len = (int)idx_len.sum().item<int64_t>();
  const bool do_compact = enable_compact && (E > 1) && (total_len >= 2048);

  if (root_fastpath && E > 1) {
    // Root fast-path: eras sorted; fabricate off_era from era_ends and identity idx_out
    auto off_era = torch::empty({ K, E + 1 }, idx_len.options());
    {
      // Build on CPU (E is small), then move to device
      auto off_cpu  = off_era.cpu();
      auto ends_cpu = era_ends.cpu();
      for (int c = 0; c < K; ++c) {
        off_cpu.data_ptr<int32_t>()[ (size_t)c*(E+1) + 0 ] = 0;
        for (int e = 0; e < E; ++e)
          off_cpu.data_ptr<int32_t>()[ (size_t)c*(E+1) + (e+1) ] = ends_cpu.data_ptr<int32_t>()[e];
      }
      off_era = off_cpu.to(bin_idx.device(), /*non_blocking=*/true);
    }
    // Identity idx_out per class (NOTE: for very large K*N you may want an implicit-index kernel variant)
    auto idx_row = torch::arange(N, torch::dtype(torch::kInt32).device(bin_idx.device()));
    auto idx_out = idx_row.unsqueeze(0).expand({ K, N }).contiguous();

    dim3 grid(k, E, ceil_div_int(K, wpb));
    dim3 blk(32 * wpb, 1, 1);
    cudaFuncSetAttribute(_h_des_mc_bfly, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_hist);
    _h_des_mc_bfly<<<grid, blk, smem_hist, stream.stream()>>>(
      bin_idx.data_ptr<int8_t>(),
      G.data_ptr<float>(), H.data_ptr<float>(),
      feat_idx.data_ptr<int32_t>(),
      idx_out.data_ptr<int32_t>(),
      off_era.data_ptr<int32_t>(),
      GH.data_ptr<float>(), HH.data_ptr<float>(),
      N, F, K, E, k, B, /*Mmax=*/N, PAD);
  }
  else if (do_compact) {
    // One-pass compaction (butterfly), then tiny scan, then histogram
    const int threads = 256;
    const int rpt = 2; // rows_per_thread (tune 2 or 4)
    const int rows_per_block = threads * rpt;
    const int max_len = (int)idx_len.max().item<int64_t>();
    const int tiles_y = max(1, ceil_div_int(max_len, rows_per_block));

    auto heads   = torch::zeros({ K, E }, idx_len.options()); // will hold counts
    auto idx_out = torch::empty({ K, Mmax }, idx_mat.options());

    // (1) compaction
    {
      dim3 grid_c(K, tiles_y, 1);
      _et_compact_bfly_onepass<<<grid_c, threads, 0, stream.stream()>>>(
        idx_mat.data_ptr<int32_t>(), idx_len.data_ptr<int32_t>(),
        era_of_row.data_ptr<int32_t>(),
        K, Mmax, E,
        heads.data_ptr<int32_t>(),
        idx_out.data_ptr<int32_t>(),
        rpt);
    }

    // (2) scan heads -> off_era
    auto off_era = torch::empty({ K, E + 1 }, idx_len.options());
    {
      dim3 grid_s(K, 1, 1);
      _scan_counts_to_offsets<<<grid_s, 1, 0, stream.stream()>>>(
        heads.data_ptr<int32_t>(), K, E,
        off_era.data_ptr<int32_t>());
    }

    // (3) histogram
    {
      dim3 grid_h(k, E, ceil_div_int(K, wpb));
      dim3 blk_h(32 * wpb, 1, 1);
      cudaFuncSetAttribute(_h_des_mc_bfly, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_hist);
      _h_des_mc_bfly<<<grid_h, blk_h, smem_hist, stream.stream()>>>(
        bin_idx.data_ptr<int8_t>(),
        G.data_ptr<float>(), H.data_ptr<float>(),
        feat_idx.data_ptr<int32_t>(),
        idx_out.data_ptr<int32_t>(),
        off_era.data_ptr<int32_t>(),
        GH.data_ptr<float>(), HH.data_ptr<float>(),
        N, F, K, E, k, B, Mmax, PAD);
    }
  }
  else {
    // Small node / single era fallback: fabricate trivial off_era per class and reuse idx_mat
    auto off_era = torch::empty({ K, E + 1 }, idx_len.options());
    {
      auto off_cpu = off_era.cpu();
      if (E == 1) {
        for (int c = 0; c < K; ++c) {
          off_cpu.data_ptr<int32_t>()[ (size_t)c*2 + 0 ] = 0;
          off_cpu.data_ptr<int32_t>()[ (size_t)c*2 + 1 ] = idx_len.cpu().data_ptr<int32_t>()[c];
        }
      } else {
        // If multiple eras but tiny node, approximate by using era_ends (safe if idx_mat built from current node)
        auto ends_cpu = era_ends.cpu();
        for (int c = 0; c < K; ++c) {
          off_cpu.data_ptr<int32_t>()[ (size_t)c*(E+1) + 0 ] = 0;
          for (int e = 0; e < E; ++e)
            off_cpu.data_ptr<int32_t>()[ (size_t)c*(E+1) + (e+1) ] = ends_cpu.data_ptr<int32_t>()[e];
        }
      }
      off_era = off_cpu.to(bin_idx.device(), /*non_blocking=*/true);
    }

    auto idx_out = idx_mat; // already (ragged) but ranges are small; acceptable

    dim3 grid(k, E, ceil_div_int(K, wpb));
    dim3 blk(32 * wpb, 1, 1);
    cudaFuncSetAttribute(_h_des_mc_bfly, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_hist);
    _h_des_mc_bfly<<<grid, blk, smem_hist, stream.stream()>>>(
      bin_idx.data_ptr<int8_t>(),
      G.data_ptr<float>(), H.data_ptr<float>(),
      feat_idx.data_ptr<int32_t>(),
      idx_out.data_ptr<int32_t>(),
      off_era.data_ptr<int32_t>(),
      GH.data_ptr<float>(), HH.data_ptr<float>(),
      N, F, K, E, k, B, Mmax, PAD);
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "h_des_mc launch failed: ", cudaGetErrorString(cudaGetLastError()));

  return torch::stack({ GH, HH }, 0); // [2, E, k, K, B]
}