cmake_minimum_required(VERSION 3.23)
project(RadixSort CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(RadixSort main.cu)

set_target_properties(RadixSort PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
