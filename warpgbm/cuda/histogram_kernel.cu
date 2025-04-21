#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define F_TILE 128 // Number of features processed per block (tile)

// Each block processes a tile of features (of size up to F_TILE) and a chunk of samples.
__global__ void histogram_kernel_shared_sample(
    const int8_t *__restrict__ bin_indices, // [N, F] bin indices
    const float *__restrict__ gradients,    // [N] gradient values
    float *__restrict__ grad_hist,          // [F * B] global gradient histogram (flattened)
    float *__restrict__ hess_hist,          // [F * B] global hessian histogram (flattened)
    int64_t N, int64_t F, int64_t B)
{
    // Use dynamic shared memory to hold the histogram for a tile.
    // Allocate 2 arrays: one for gradients and one for hessians.
    extern __shared__ float shmem[];
    float *shared_grad = shmem;                // size: tile_features * B floats
    float *shared_hess = shmem + (F_TILE * B); // same size

    int tid = threadIdx.x; // Use a 1D block (for sample processing)
    int block_size = blockDim.x;

    // Each block is assigned a tile of features:
    int feature_offset = blockIdx.x * F_TILE;
    // Adjust tile width if we're near the end of the feature dimension.
    int tile_features = (feature_offset + F_TILE > F) ? (F - feature_offset) : F_TILE;
    int tile_size = tile_features * B; // total number of bins in this feature tile

    // Initialize the tile’s shared memory histograms.
    for (int i = tid; i < tile_size; i += block_size)
    {
        shared_grad[i] = 0.0f;
        shared_hess[i] = 0.0f;
    }
    __syncthreads();

    // Each block also covers a chunk of samples. Determine the sample index
    int sample = blockIdx.y * block_size + tid;
    if (sample < N)
    {
        // For each feature in this tile, compute the bin and update shared histograms.
        for (int j = 0; j < tile_features; j++)
        {
            // Global feature index.
            int f_idx = feature_offset + j;
            int64_t idx = sample * F + f_idx; // index into the [N, F] bin_indices tensor
            int8_t b = bin_indices[idx];      // get bin index
            if (b >= 0 && b < B)
            {
                int shared_idx = j * B + b; // index into the tile histogram in shared memory
                // Using atomics because several threads may update the same bin.
                atomicAdd(&shared_grad[shared_idx], gradients[sample]);
                atomicAdd(&shared_hess[shared_idx], 1.0f);
            }
        }
    }
    __syncthreads();

    // Flush the per-tile histograms from shared memory to global memory.
    // Each bin in the tile is added to the global histogram (which is sized [F, B]).
    for (int i = tid; i < tile_size; i += block_size)
    {
        int local_feature = i / B; // feature index relative to the tile
        int bin = i % B;           // bin index
        int f_idx = feature_offset + local_feature;
        if (f_idx < F)
        {
            int global_idx = f_idx * B + bin;
            atomicAdd(&grad_hist[global_idx], shared_grad[i]);
            atomicAdd(&hess_hist[global_idx], shared_hess[i]);
        }
    }
}

void launch_histogram_kernel_cuda(
    const at::Tensor &bin_indices, // [N, F] int8 tensor
    const at::Tensor &gradients,   // [N] float tensor
    at::Tensor &grad_hist,         // [F * B] float tensor (preallocated)
    at::Tensor &hess_hist,         // [F * B] float tensor (preallocated)
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t B = num_bins;

    // Define grid and block dimensions.
    // blockDim.x: number of threads per block (for processing samples).
    // gridDim.x: number of feature tiles.
    int grid_x = (F + F_TILE - 1) / F_TILE;
    // gridDim.y: number of sample chunks.
    int grid_y = (N + threads_per_block - 1) / threads_per_block;
    dim3 blocks(grid_x, grid_y);
    dim3 threads(threads_per_block);

    // Calculate shared memory size:
    // We allocate 2 arrays of size (F_TILE * B) floats (one for grad and one for hess).
    size_t shared_mem_size = 2 * F_TILE * B * sizeof(float);

    histogram_kernel_shared_sample<<<blocks, threads, shared_mem_size>>>(
        bin_indices.data_ptr<int8_t>(),
        gradients.data_ptr<float>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, F, B);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// CUDA kernel: tiled, 64-bit safe
__global__ void histogram_tiled_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F]
    const float *__restrict__ gradients,    // [N]
    float *__restrict__ grad_hist,          // [F * B]
    float *__restrict__ hess_hist,          // [F * B]
    int64_t F, int64_t B, int64_t tile_size)
{
    int64_t feature_tiles = (F + tile_size - 1) / tile_size;
    int64_t row = static_cast<int64_t>(blockIdx.x) / feature_tiles;
    int64_t tile = static_cast<int64_t>(blockIdx.x) % feature_tiles;
    int64_t feat = tile * tile_size + threadIdx.x;

    if (feat >= F)
        return;

    int8_t bin = bin_indices[row * F + feat];
    if (bin >= 0 && bin < B)
    {
        int64_t idx = feat * B + bin;
        atomicAdd(&grad_hist[idx], gradients[row]);
        atomicAdd(&hess_hist[idx], 1.0f);
    }
}

// Host function exposed to PyTorch
void launch_histogram_kernel_cuda_2(
    const at::Tensor &bin_indices, // int8 [N, F]
    const at::Tensor &gradients,   // float32 [N]
    at::Tensor &grad_hist,         // float32 [F * B]
    at::Tensor &hess_hist,         // float32 [F * B]
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    CHECK_INPUT(bin_indices);
    CHECK_INPUT(gradients);
    CHECK_INPUT(grad_hist);
    CHECK_INPUT(hess_hist);

    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t tile_size = threads_per_block;
    int64_t feature_tiles = (F + tile_size - 1) / tile_size;
    int64_t total_blocks = N * feature_tiles;

    histogram_tiled_kernel<<<
        static_cast<int>(total_blocks),
        static_cast<int>(tile_size)>>>(
        bin_indices.data_ptr<int8_t>(),
        gradients.data_ptr<float>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        F, num_bins, tile_size);

    // Optional: check for kernel launch failure
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void histogram_tiled_configurable_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F]
    const float *__restrict__ gradients,    // [N]
    float *__restrict__ grad_hist,          // [F * B]
    float *__restrict__ hess_hist,          // [F * B]
    int64_t N, int64_t F, int64_t B,
    int rows_per_thread)
{
    int feat = blockIdx.x; // 1 block per feature
    int row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

    extern __shared__ float shmem[];
    float *sh_grad = shmem;       // [B]
    float *sh_hess = &sh_grad[B]; // [B]

    // Initialize shared memory histograms
    for (int b = threadIdx.x; b < B; b += blockDim.x)
    {
        sh_grad[b] = 0.0f;
        sh_hess[b] = 0.0f;
    }
    __syncthreads();

    // Each thread processes multiple rows
    for (int r = 0; r < rows_per_thread; ++r)
    {
        int row = row_start + r;
        if (row < N)
        {
            int8_t bin = bin_indices[row * F + feat];
            if (bin >= 0 && bin < B)
            {
                atomicAdd(&sh_grad[bin], gradients[row]);
                atomicAdd(&sh_hess[bin], 1.0f);
            }
        }
    }
    __syncthreads();

    // One thread per bin writes results back to global memory
    for (int b = threadIdx.x; b < B; b += blockDim.x)
    {
        int64_t idx = feat * B + b;
        atomicAdd(&grad_hist[idx], sh_grad[b]);
        atomicAdd(&hess_hist[idx], sh_hess[b]);
    }
}

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &gradients,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    CHECK_INPUT(bin_indices);
    CHECK_INPUT(gradients);
    CHECK_INPUT(grad_hist);
    CHECK_INPUT(hess_hist);

    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);

    int rows_per_block = threads_per_block * rows_per_thread;
    int row_tiles = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles); // grid.x = F, grid.y = row_tiles
    dim3 threads(threads_per_block);
    int shared_mem_bytes = 2 * num_bins * sizeof(float);

    histogram_tiled_configurable_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        gradients.data_ptr<float>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, F, num_bins,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void fused_histogram_split_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F]
    const float *__restrict__ gradients,    // [N]
    const float *__restrict__ parent_grad,  // [F * B]
    const float *__restrict__ parent_hess,  // [F * B]
    float *__restrict__ grad_hist,          // [F * B]
    float *__restrict__ hess_hist,          // [F * B]
    float *__restrict__ best_gain_first,    // [F]
    int *__restrict__ best_bin_first,       // [F]
    float *__restrict__ best_gain_second,   // [F]
    int *__restrict__ best_bin_second,      // [F]
    int64_t N, int64_t F, int64_t B,
    float min_child_weight,
    float min_split_gain,
    float eps,
    int rows_per_thread)
{
    int feat = blockIdx.x;
    int tid = threadIdx.x;
    int row_start = (blockIdx.y * blockDim.x + tid) * rows_per_thread;

    extern __shared__ float shmem[];
    float *final_grad = shmem;     // [B]
    float *final_hess = &shmem[B]; // [B]

    for (int b = tid; b < B; b += blockDim.x)
    {
        final_grad[b] = 0.0f;
        final_hess[b] = 0.0f;
    }
    __syncthreads();

    // Accumulate local histogram
    for (int r = 0; r < rows_per_thread; ++r)
    {
        int row = row_start + r;
        if (row < N)
        {
            int8_t bin = bin_indices[row * F + feat];
            if (bin >= 0 && bin < B)
            {
                atomicAdd(&final_grad[bin], gradients[row]);
                atomicAdd(&final_hess[bin], 1.0f);
            }
        }
    }
    __syncthreads();

    for (int b = tid; b < B; b += blockDim.x)
    {
        int64_t idx = feat * B + b;
        atomicAdd(&grad_hist[idx], final_grad[b]);
        atomicAdd(&hess_hist[idx], final_hess[b]);
    }
    __syncthreads();

    // Only thread 0 of each block may proceed to check the global state
    if (tid < 2)
    {
        float global_hess_sum = 0.0f;
        for (int b = 0; b < B; ++b)
        {
            global_hess_sum += hess_hist[feat * B + b];
        }

        if (fabsf(global_hess_sum - static_cast<float>(N)) > eps)
        {
            // printf("❗Feature %d: global_hess_sum = %.1f vs N = %lld → SKIP\n", feat, global_hess_sum, N);
            return; // Don't proceed until histogram is fully accumulated
        }

        float local_grad[128] = {0};
        float local_hess[128] = {0};

        for (int b = 0; b < B; ++b)
        {
            float g_hist = grad_hist[feat * B + b];
            float h_hist = hess_hist[feat * B + b];
            float g_parent = parent_grad[feat * B + b];
            float h_parent = parent_hess[feat * B + b];

            // For tid == 0: compute left child (histogram data)
            // For tid == 1: compute right child (parent - histogram)
            if (tid == 0)
            {
                local_grad[b] = g_hist;
                local_hess[b] = h_hist;
            }
            else
            {
                local_grad[b] = g_parent - g_hist;
                local_hess[b] = h_parent - h_hist;
            }
        }

        float G_total = 0.0f, H_total = 0.0f;
        for (int b = 0; b < B; ++b)
        {
            G_total += local_grad[b];
            H_total += local_hess[b];
        }

        float G_L = 0.0f, H_L = 0.0f;
        float best_gain = min_split_gain;
        int best_bin = -1;

        for (int b = 0; b < B - 1; ++b)
        {
            G_L += local_grad[b];
            H_L += local_hess[b];
            float G_R = G_total - G_L;
            float H_R = H_total - H_L;

            // if (H_L < min_child_weight || H_R < min_child_weight)
            //     printf("Feature %d Bin %d: H_L = %.2f, H_R = %.2f — Skipped due to min_child_weight\n", feat, b, H_L, H_R);
            // else
            //     printf("Feature %d Bin %d: H_L = %.2f, H_R = %.2f — gain = %.5f\n", feat, b, H_L, H_R, (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps) - (G_total * G_total) / (H_total + eps));

            if (H_L >= min_child_weight && H_R >= min_child_weight)
            {
                float gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps) - (G_total * G_total) / (H_total + eps);
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_bin = b;
                }
            }
        }

        // printf("Feature %d (tid=%d): Best bin = %d, gain = %.5f\n", feat, tid, best_bin, best_gain);
        if (tid == 0)
        {
            best_gain_first[feat] = best_gain;
            best_bin_first[feat] = best_bin;
        }
        else if (tid == 1)
        {
            best_gain_second[feat] = best_gain;
            best_bin_second[feat] = best_bin;
        }
    }
}

void launch_fused_histogram_split_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &gradients,
    const at::Tensor &parent_grad,
    const at::Tensor &parent_hess,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    at::Tensor &best_gain_first,
    at::Tensor &best_bin_first,
    at::Tensor &best_gain_second,
    at::Tensor &best_bin_second,
    int num_bins,
    float min_child_weight,
    float min_split_gain,
    float eps,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    CHECK_INPUT(bin_indices);
    CHECK_INPUT(gradients);
    CHECK_INPUT(parent_grad);
    CHECK_INPUT(parent_hess);
    CHECK_INPUT(grad_hist);
    CHECK_INPUT(hess_hist);
    CHECK_INPUT(best_gain_first);
    CHECK_INPUT(best_bin_first);
    CHECK_INPUT(best_gain_second);
    CHECK_INPUT(best_bin_second);

    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);

    // if (N == 0)
    // {
    //     std::cerr << "[WarpGBM Fused Kernel Warning] Launch skipped because N == 0 (no rows to process).\n";
    //     std::cerr << "  bin_indices.size(0): " << N << ", bin_indices.size(1): " << F << std::endl;
    //     std::cerr << "  This should not happen. Investigate upstream logic.\n";
    //     return;
    // }

    int rows_per_block = threads_per_block * rows_per_thread;
    int row_tiles = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles);
    dim3 threads(threads_per_block);
    int shared_mem_bytes = 2 * num_bins * sizeof(float);

    // std::cout << "Launching fused kernel with config:" << std::endl;
    // std::cout << "  Threads per block: " << threads_per_block << std::endl;
    // std::cout << "  Rows per thread: " << rows_per_thread << std::endl;
    // std::cout << "  Grid: (" << F << ", " << row_tiles << ")" << std::endl;
    // std::cout << "  Shared memory: " << 2 * num_bins * sizeof(float) << " bytes" << std::endl;

    fused_histogram_split_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        gradients.data_ptr<float>(),
        parent_grad.data_ptr<float>(),
        parent_hess.data_ptr<float>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        best_gain_first.data_ptr<float>(),
        best_bin_first.data_ptr<int>(),
        best_gain_second.data_ptr<float>(),
        best_bin_second.data_ptr<int>(),
        N, F, num_bins,
        min_child_weight,
        min_split_gain,
        eps,
        rows_per_thread);
}
