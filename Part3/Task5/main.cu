#include <iostream>
#include "math.h"
#include "chrono"

__global__ void zeta(double *sum, int n, double s, int nThreads, int nBlocks){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double x;
    for(int i = index; i < n; i += nThreads * nBlocks){
        x = 1.0 / pow((double) i + 1, s);
        sum[index] += x;
    }
}


int main() {
    int nThreads = 512;
    int nBlocks = 30000;
    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1, 1);
    printf("Введите число шагов: ");
    int n = 1000000;
    std::cin >> n;
    printf("Введите s: ");
    double s = 2.0;
    std::cin >> s;
    size_t sum_size = nThreads * nBlocks * sizeof(double);
    double *sum_local = (double*) malloc(sum_size);
    double *sum_dev;
    cudaMalloc((void**) &sum_dev, sum_size);
    cudaMemset(sum_dev, 0, sum_size);
    auto start = std::chrono::high_resolution_clock::now();
    zeta<<<dimGrid, dimBlock>>>(sum_dev, n, s, nThreads, nBlocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s ",
                                   cudaGetErrorString(err));
    cudaMemcpy(sum_local, sum_dev, sum_size, cudaMemcpyDeviceToHost);
    double func = 0.0;
    for(int i = 0; i < nThreads * nBlocks; i ++){
        func += sum_local[i];
    }
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Значение дзета функции Римана: %.20f", func);
    printf("Time ");
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf(" ms\n");
    cudaFree(sum_dev);
    free(sum_local);
    return 0;
}
