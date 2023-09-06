cmake_minimum_required(VERSION 3.17)
project(Task1 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Task1 main.cu)

set_target_properties(
        Task1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)