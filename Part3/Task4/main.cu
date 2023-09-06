#include <iostream>
#include "math.h"
#include "chrono"

__global__ void integral(double *sum, int n, double step, int nThreads, int nBlocks){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double x;
    for(int i = index; i < n; i += nThreads * nBlocks){
        x = i * step;
        sum[index] += 4.0 / (1.0 + x * x);
    }
}

int main() {
    int nThreads = 512;
    int nBlocks = 65000;
    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1, 1);
    int n = 100000000000;
    double step = 1.0 / n;
    size_t size = nThreads * nBlocks * sizeof(double);
    double *sum = (double*) malloc(size);
    double *sumDev;
    cudaMalloc((void**) &sumDev, size);
    cudaMemset(sumDev, 0, size);
    auto start = std::chrono::high_resolution_clock::now();
    integral<<<dimGrid, dimBlock>>>(sumDev, n, step, nThreads, nBlocks);
    cudaMemcpy(sum, sumDev, size, cudaMemcpyDeviceToHost);
    double pi = 0.0;
    for(int i = 0; i < nThreads*nBlocks; i++){
        pi += sum[i];
    }
    pi *= step;
    auto end = std::chrono::high_resolution_clock::now();
    double diff = abs(pi - M_PI);
    printf("Pi = %.15f\n", pi);
    printf("Diff = %.15f\n", diff);
    printf("Time: ");
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf(" ms \n");
    free(sum);
    cudaFree(sumDev);
    return 0;
}
