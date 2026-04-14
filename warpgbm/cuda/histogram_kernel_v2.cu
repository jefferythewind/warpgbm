#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template<int B>
__device__ __forceinline__ constexpr int choose_t() {
    // heuristic: keep ~128 threads when possible
    const int q = 128 / B; // for B in [2..127], q in [1..64]
    return (q >= 32) ? 32 :
           (q >= 16) ? 16 :
           (q >=  8) ?  8 :
           (q >=  4) ?  4 :
           (q >=  2) ?  2 : 1;
}


template <int64_t B>
__global__ void compute_histograms(
    const int8_t *__restrict__ bin_indices, // [N, F_master]
    const float *__restrict__ residuals,    // [N]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N]
    float *__restrict__ grad_hist,          // [F * B]
    float *__restrict__ hess_hist,          // [F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t E
)
{
    constexpr int64_t t = choose_t<B>();
    const int64_t n_base = (((blockIdx.x * blockDim.x) + threadIdx.x) >> 5) << 7;
    const int fid = blockIdx.y;
    const int curr_era = blockIdx.z;

    if (fid >= F) return;

    const int64_t mfid = (int64_t)feature_indices[fid];

    if (mfid >= F_master) return;

    __shared__ float2 ghv2_smem[B];

    for (int i = threadIdx.x; i < B; i+=blockDim.x) {

        ghv2_smem[i] = make_float2(0.0f, 0.0f);
    }

    __syncthreads();

    const int lane = threadIdx.x & 31;
    
    float2 ghv2 = make_float2(0.0f, 0.0f);

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        int64_t nid = n_base + (i << 5) + ((lane / t) + (lane & (t - 1)));
        
        if (nid < N) {
            int64_t sample = (int64_t)sample_indices[nid];
            
            if (era_indices[sample] == curr_era) {
                
                if (bin_indices[sample * F_master + mfid] == (lane & (t - 1))) {
                    ghv2.x += residuals[sample];
                    ghv2.y += 1.0f;
                }
            }
        }
    }

    const unsigned mask = (t == 32) ? 0xFFFFFFFFu : (((1u << t) - 1u) << ((lane / t) * t));


    for (int i = 0; i < t; ++t) {

        float2 tmp;
        tmp.x = __shfl_xor_sync(mask, ghv2.x, i ^ (lane & (t - 1)), t);
        tmp.y = __shfl_xor_sync(mask, ghv2.y, i ^ (lane & (t - 1)), t);
        ghv2.x += tmp.x;
        ghv2.y += tmp.y;

    }


    for (int i = threadIdx.x; i < B; i+=blockDim.x) {

        if (ghv2.y > 0.0f) {
            atomicAdd(&ghv2_smem[i].x, ghv2.x);
            atomicAdd(&ghv2_smem[i].y, ghv2.y);
        }
    }

    __syncthreads();


    if (threadIdx.x < B) {
        int64_t global_idx = (curr_era * F * B) + (fid * B) + threadIdx.x;
        if (ghv2.y > 0.0f) {
            atomicAdd(&grad_hist[global_idx], ghv2_smem[threadIdx.x].x);
            atomicAdd(&hess_hist[global_idx], ghv2_smem[threadIdx.x].y);
        }
    }
}



void launch_histogram_kernel_cuda_v2(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 512,
    int rows_per_thread = 1)
{

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int num_features_master = bin_indices.size(1);

    int64_t rows_per_block = threads_per_block * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles); // grid.x = F, grid.y = row_tiles
    dim3 threads(threads_per_block);
    int num_eras = grad_hist.size(0); // inferred from output tensor
    int shared_mem_bytes = 2 * num_eras * num_bins * sizeof(float);

    histogram_tiled_configurable_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, num_features_master, F, num_bins, num_eras,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}