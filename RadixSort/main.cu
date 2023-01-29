#include <iostream>
#include <vector>

__host__ void radix_sort(std::vector<uint32_t> data) {
    const int len = data.size();

    uint32_t cpu_tmp_0[len];
    uint32_t cpu_tmp_1[len];
    for (int bit = 0; bit < 32; bit++) {
        uint32_t x = 0;
        uint32_t y = 0;
        for (int i = 0; i < data.size(); i++) {
            if ((data[i] & (1 << bit)) == 0) {
                cpu_tmp_0[x++] = data[i];
            } else {
                cpu_tmp_1[y++] = data[i];
            }
        }

        //合并
        for (int i = 0; i < x; i++) {
            data[i] = cpu_tmp_0[i];
        }
        for (int i = 0; i < y; i++) {
            data[x + i] = cpu_tmp_1[i];
        }

    }
}

//gpu
__device__ void radix_sort_gpu(std::vector<uint32_t> data, const uint32_t tid, const uint32_t num_threads) {
    const int len = data.size();

    uint32_t cpu_tmp_0[len];
    uint32_t cpu_tmp_1[len];
    for (int bit = 0; bit < 32; bit++) {
        uint32_t x = 0;
        uint32_t y = 0;
        for (int i = 0; i < data.size(); i += num_threads) {
            if ((data[i] & (1 << bit)) == 0) {
                cpu_tmp_0[x] = data[i + tid];
                x += num_threads;
            } else {
                cpu_tmp_1[y] = data[i + tid];
                y += num_threads;
            }
        }

        //合并
        for (int i = 0; i < x; i += num_threads) {
            data[i + tid] = cpu_tmp_0[i + tid];
        }
        for (int i = 0; i < y; i += num_threads) {
            data[x + i + tid] = cpu_tmp_1[i + tid];
        }

    }
    __syncthreads();
}

__global__ void sort(std::vector<int> &data){
    radix_sort_gpu(data,threadIdx.x, blockDim.x * blockIdx.x)
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
