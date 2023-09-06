#include <iostream>
#include "ctime"
#include "chrono"

__global__ void scalar(double *x, double *y, double *sum, int n, int nThreads, int nBlocks){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    double m;
    for (int i = idx; i < n; i += nThreads * nBlocks){
        m = x[i] * y[i];
        sum[idx] += m;
    }
}

int main() {
    std::srand(time(nullptr));
    int n = 100000;
    int nThreads = 960;
    int nBlocks = 100;
    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1,1);
    double x[n];
    double y[n];
    for(int i = 0; i < n; i ++){
        x[i] = ((double) (std::rand() % 100000001) / 100000) - 500;
        y[i] = ((double) (std::rand() % 100000001) / 100000) - 500;
    }
    auto start = std::chrono::high_resolution_clock::now();
    size_t array_size = sizeof(double) * n;
    size_t return_size = sizeof(double) * nThreads * nBlocks;
    double *x_dev;
    double *y_dev;
    double *sum_dev;
    cudaMalloc((void **) &x_dev, array_size);
    cudaMalloc((void **) &y_dev, array_size);
    cudaMalloc((void **) &sum_dev, return_size);
    cudaMemset(sum_dev, 0.0, return_size);
    cudaMemcpy(x_dev, x, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y, array_size, cudaMemcpyHostToDevice);
    scalar<<<dimGrid, dimBlock>>>(x_dev, y_dev, sum_dev, n, nThreads, nBlocks);
    cudaDeviceSynchronize();
    double *sum = (double*) malloc(return_size);
    cudaMemcpy(sum, sum_dev, return_size, cudaMemcpyDeviceToHost);
    double scalar_total = 0;
    for(int i = 0; i < nThreads * nBlocks; i ++){
        scalar_total += sum[i];
    }
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Scalar sum = %.3f\n", scalar_total);
    printf("Time: ");
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf(" ms\n");
    cudaFree(sum_dev);
    cudaFree(y_dev);
    cudaFree(x_dev);
    free(sum);
    return 0;
}
