cmake_minimum_required(VERSION 3.17)
project(Task4 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Task4 main.cu)

set_target_properties(
        Task4
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)