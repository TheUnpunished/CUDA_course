cmake_minimum_required(VERSION 3.17)
project(Task1_3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Task1_3 main.cu)

set_target_properties(
        Task1_3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)