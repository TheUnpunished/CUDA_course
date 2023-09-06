cmake_minimum_required(VERSION 3.17)
project(Task3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Task3 main.cu)

set_target_properties(
        Task3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)