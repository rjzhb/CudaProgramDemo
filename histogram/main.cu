
//最低效的方法
__global__ void myhistogram256kernel_02(
        const unsigned int const *d_hist_data,
        unsigned int *const d_bin_data
) {
    const unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const unsigned int idy = (blockDim.y * blockIdx.y) + threadIdx.y;
    const unsigned int tid = (gridDim.x * blockDim.x * idy) + idx;
    const unsigned char value = d_hist_data[tid];

    atomicAdd(&(d_bin_data[value]), 1);
}

//第一级优化,一下子读写四个字节(四个直方图元素)，但实际上GTX460这种级别的卡以后就会自动做这种优化了，没意义
__global__ void myhistogram256kernel_03(
        const unsigned int const *d_hist_data,
        unsigned int *const d_bin_data
) {
    const unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const unsigned int idy = (blockDim.y * blockIdx.y) + threadIdx.y;
    const unsigned int tid = (gridDim.x * blockDim.x * idy) + idx;
    const unsigned int value_u32 = d_hist_data[tid];

    atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF))]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00))]), 8);
    atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000))]), 16);
    atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000))]), 24);
}

//第二级优化,使用共享内存配合同步，将写次数合并

__shared__ unsigned int d_bin_data_shared[256];

__global__ void myhistogram256kernel_04(
        const unsigned int const *d_hist_data,
        unsigned int *const d_bin_data
) {
    const unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const unsigned int idy = (blockDim.y * blockIdx.y) + threadIdx.y;
    const unsigned int tid = (gridDim.x * blockDim.x * idy) + idx;
    //读操作：
    d_bin_data_shared[threadIdx.x] = 0;
    const unsigned int value_u32 = d_hist_data[tid];
    __syncthreads();
    //写入共享内存
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0x000000FF))]), 1);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0x0000FF00))]), 8);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0x00FF0000))]), 16);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0xFF000000))]), 24);
    __syncthreads();
    //一次性从共享内存写入全局内存
    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}
