#include <iostream>
#include <vector>

#define MAX_NUM_LISTS 1024
#define NUM_ELEM 6

__host__ void radix_sort(std::vector<uint32_t> &data) {
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
__device__ void radix_sort_gpu(uint32_t *const sort_tmp,
                               const uint32_t num_elements,
                               const uint32_t tid,
                               const uint32_t num_threads,
                               uint32_t *const sort_tmp_1
) {
    for (int bit = 0; bit < 32; bit++) {
        uint32_t x = 0;
        uint32_t y = 0;
        for (int i = 0; i < num_elements; i += num_threads) {
            if ((sort_tmp[i] & (1 << bit)) == 0) {
                sort_tmp[x] = sort_tmp[i + tid];
                x += num_threads;
            } else {
                sort_tmp_1[y] = sort_tmp[i + tid];
                y += num_threads;
            }
        }

        //合并
        for (int i = 0; i < y; i += num_threads) {
            sort_tmp[x + i + tid] = sort_tmp_1[i + tid];
        }

    }
    __syncthreads();
}

uint32_t find_min(
        const uint32_t *const src_array,
        uint32_t *const list_indexes,
        const uint32_t num_lists,
        const uint32_t num_elements_per_list
) {
    uint32_t min_val = 0XFFFFFFFF;
    uint32_t min_idx = 0;

    for (uint32_t i = 0; i < num_lists; i++) {
        if (list_indexes[i] < num_elements_per_list) {
            const uint32_t src_idx = i + (list_indexes[i] * num_lists);
            const uint32_t data = src_array[src_idx];
            if (data <= min_val) {
                min_val = data;
                min_idx = 1;
            }
        }
    }
    list_indexes[min_idx]++;
    return min_val;
}

//单线程合并
void merge_array(
        const uint32_t *const src_array,
        uint32_t *const dest_array,
        const uint32_t num_lists,
        const uint32_t num_elements
) {
    const uint32_t num_elements_per_list = num_elements / num_lists;
    uint32_t list_indexes[MAX_NUM_LISTS];

    for (uint32_t list = 0; list < num_lists; list++) {
        list_indexes[list] = 0;
    }

    for (uint32_t i = 0; i < num_elements; i++) {
        dest_array[i] = find_min(src_array, list_indexes, num_lists, num_elements_per_list);
    }
}

//并行合并
__device__ void merge_array_parallel(
        const uint32_t *const src_array,
        uint32_t *const dest_array,
        const uint32_t num_lists,
        const uint32_t num_elements,
        const uint32_t tid
) {
    const uint32_t num_elements_per_list = num_elements / num_lists;
    __shared__ uint32_t list_indexes[MAX_NUM_LISTS];
    //clear
    list_indexes[tid] = 0;

    __syncthreads();

    for (uint32_t i = 0; i < num_elements; i++) {
        __shared__ uint32_t min_val;
        __shared__ uint32_t min_tid;

        uint32_t data;
        if (list_indexes[tid] < num_elements_per_list) {
            const uint32_t src_idx = tid + (list_indexes[tid] * num_lists);
            data = src_array[src_idx];
        } else {
            data = 0xFFFFFFFF;
        }

        //让线程0负责清空共享内存这块数据
        if (tid == 0) {
            min_val = 0xFFFFFFFF;
            min_tid = 0xFFFFFFFF;
        }

        __syncthreads();

        //各个线程相互对比
        atomicMin(&min_val, data);

        __syncthreads();

        if (min_val == data) {
            atomicMin(&min_tid, tid);
        }
        __syncthreads();

        if (tid == min_tid) {
            list_indexes[tid]++;
            dest_array[i] = data;
        }
    }
}

//并行归约型合并
__device__ void merge_array_parallel2(
        const uint32_t *const src_array,
        uint32_t *const dest_array,
        const uint32_t num_lists,
        const uint32_t num_elements,
        const uint32_t tid
) {
    const uint32_t num_elements_per_list = num_elements / num_lists;
    __shared__ uint32_t list_indexes[MAX_NUM_LISTS];
    __shared__ uint32_t reduction_val[MAX_NUM_LISTS];
    __shared__ uint32_t reduction_idx[MAX_NUM_LISTS];
    //clear
    list_indexes[tid] = 0;
    reduction_idx[tid] = 0;
    reduction_val[tid] = 0;
    __syncthreads();

    for (uint32_t i = 0; i < num_elements; i++) {
        uint32_t tid_max = num_lists >> 1;
        uint32_t data;
        if (list_indexes[tid] < num_elements_per_list) {
            const uint32_t src_idx = tid + (list_indexes[tid] * num_lists);
            data = src_array[src_idx];
        } else {
            data = 0xFFFFFFFF;
        }

        reduction_val[tid] = data;
        reduction_idx[tid] = tid;

        __syncthreads();

        while (tid_max != 0) {
            if (tid < tid_max) {
                const uint32_t val2_idx = tid + tid_max;
                const uint32_t val2 = reduction_val[val2_idx];


                if (reduction_val[tid] > val2) {
                    reduction_val[tid] = val2;
                    reduction_idx[tid] = reduction_idx[val2_idx];
                }
            }

            tid_max >>= 1;
            __syncthreads();
        }

        if (tid == 0) {
            list_indexes[reduction_idx[0]]++;
            dest_array[i] = reduction_val[0];
        }
        __syncthreads();
    }
}

__device__ void copy_data_to_shared(
        const uint32_t *const data,
        uint32_t *const sort_tmp,
        const uint32_t num_lists,
        const uint32_t num_elements,
        const uint32_t tid
) {
    for (uint32_t i = 0; i < num_elements; i += num_lists) {
        sort_tmp[i + tid] = data[i + tid];
    }
    __syncthreads();
}

__global__ void sort(
        uint32_t *const data,
        const uint32_t num_lists,
        const uint32_t num_elements
) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ uint32_t sort_tmp[NUM_ELEM];
    __shared__ uint32_t sort_tmp_1[NUM_ELEM];

    copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);

    radix_sort_gpu(sort_tmp, num_elements, tid, num_lists, sort_tmp_1);

    merge_array_parallel(sort_tmp, data, num_lists, num_elements, tid);

}

