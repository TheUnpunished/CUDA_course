cmake_minimum_required(VERSION 3.17)
project(Task2_5 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Task2_5 main.cu)

set_target_properties(
        Task2_5
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)