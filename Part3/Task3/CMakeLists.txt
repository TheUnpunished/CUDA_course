cmake_minimum_required(VERSION 3.17)
project(Task3 CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories(/usr/local/cuda-8.0/samples/common/inc)

add_executable(Task3 main.cu)

set_target_properties(
        Task3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)