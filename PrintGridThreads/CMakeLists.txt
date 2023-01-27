cmake_minimum_required(VERSION 3.23)
project(PrintGridThreads CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(PrintGridThreads main.cu)

set_target_properties(PrintGridThreads PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
