cmake_minimum_required(VERSION 3.23)
project(Histogram CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Histogram main.cu)

set_target_properties(Histogram PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
