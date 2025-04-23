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

__global__ void histogram_kernel(
    const int8_t *__restrict__ bin_indices,     // [M, F]
    const float *__restrict__ gradients,        // [M]
    const int32_t *__restrict__ sample_to_node, // [M]
    float *__restrict__ grad_hist,              // [max_nodes * F * B]
    float *__restrict__ hess_hist,              // [max_nodes * F * B]
    int64_t M, int64_t F, int64_t B,
    int rows_per_thread)
{
    int feat = blockIdx.x; // 1 block per feature
    int row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

    extern __shared__ float shmem[];
    float *sh_grad = shmem;       // [B]
    float *sh_hess = &sh_grad[B]; // [B]

    for (int b = threadIdx.x; b < B; b += blockDim.x)
    {
        sh_grad[b] = 0.0f;
        sh_hess[b] = 0.0f;
    }
    __syncthreads();

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int row = row_start + r;
        if (row < M)
        {
            int node_id = sample_to_node[row];
            int8_t bin = bin_indices[row * F + feat];
            if (bin >= 0 && bin < B)
            {
                int64_t idx = node_id * F * B + feat * B + bin;
                atomicAdd(&grad_hist[idx], gradients[row]);
                atomicAdd(&hess_hist[idx], 1.0f);
            }
        }
    }
    __syncthreads();
}

void build_histograms(
    const at::Tensor &bin_indices,
    const at::Tensor &sample_to_node,
    const at::Tensor &residual,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int threads_per_block = 64,
    int rows_per_thread = 4)
{
    TORCH_CHECK(bin_indices.dim() == 2, "bin_indices must be 2D");
    int64_t M = bin_indices.size(0); // number of samples
    int64_t F = bin_indices.size(1);
    int B = grad_hist.size(2);

    int rows_per_block = threads_per_block * rows_per_thread;
    int row_tiles = (M + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles);
    dim3 threads(threads_per_block);
    int shared_mem_bytes = 2 * B * sizeof(float);

    histogram_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residual.data_ptr<float>(),
        sample_to_node.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        M, F, B, rows_per_thread);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}