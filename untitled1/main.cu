#include <iostream>

//GPU function
__global__ void set_id(unsigned int *const block,
                       unsigned int *const thread,
                       unsigned int *const warp,
                       unsigned int *const calc_thread) {
    //true id
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    //true id array
    calc_thread[thread_idx] = thread_idx;
    //block id
    block[thread_idx] = blockIdx.x;
    //id in a block
    thread[thread_idx] = threadIdx.x;
    //thread warp
    warp[thread_idx] = threadIdx.x / warpSize;
}

//CPU function
#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

int main() {
    //two blocks, per block have 64 threads
    const unsigned int num_blocks = 2;
    const unsigned int num_threads = 64;

    //GPU info
    unsigned int *gpu_block;
    unsigned int *gpu_thread;
    unsigned int *gpu_warp;
    unsigned int *gpu_calc_thread;

    //alloc to gpu
    cudaMalloc((void **) &gpu_block, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **) &gpu_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **) &gpu_warp, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **) &gpu_calc_thread, ARRAY_SIZE_IN_BYTES);

    //invoke kernel function
    set_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    //move to cpu info
    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    //Free arrays on the GPU
    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    //print
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("Calculated Thread: %3u - Block: %2u - Warp %2u - Thread %3u\n",
               cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }


    return 0;
}
