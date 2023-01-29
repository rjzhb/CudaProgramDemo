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

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
