#include <iostream>
#include "curand_kernel.h"
#include "chrono"
#include "math.h"

__device__ float my_rand(curandState state){
    return curand_uniform(&state);
}

__global__ void kernel_init(curandState *state_x, curandState *state_y, int offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock64(), idx, offset * 2, &state_x[idx]);
    curand_init(clock64(), idx, offset * 2 + 1, &state_y[idx]);
}

__global__ void methodMC(int *mCarlo, int n, int nThreads, int nBlocks,
                         curandState *state_x, curandState *state_y, int offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx + offset * nThreads * nBlocks <= n){
        float x = my_rand(state_x[idx]);
        float y = my_rand(state_y[idx]);
//        printf("%f %f \n", x, y);
        if(x * x + y * y <= 1){
            atomicAdd(mCarlo, 1);
        }
    }
}

int main() {
    int nThreads = 200;
    int nBlocks = 512;
    dim3 dimBlock(nThreads, 1,1);
    dim3 dimGrid(nBlocks, 1, 1);
    int n = 10000000;
    int * mCarlo_dev;
    cudaMalloc((void **) &mCarlo_dev, sizeof(int));
    cudaMemset(mCarlo_dev, 0, sizeof(int));
    int offset_max = n / (nBlocks * nThreads);
    if(n % (nBlocks * nThreads) != 0){
        offset_max ++;
    }
    auto start = std::chrono::high_resolution_clock::now();
    for(int offset = 0; offset < offset_max; offset ++){
        curandState *state_x;
        curandState *state_y;
        cudaMalloc((void **) &state_x, sizeof(curandState) * nThreads * nBlocks);
        cudaMalloc((void **) &state_y, sizeof(curandState) * nThreads * nBlocks);
        kernel_init<<<dimGrid, dimBlock>>>(state_x, state_y, offset);
        cudaDeviceSynchronize();
        methodMC<<<dimGrid, dimBlock>>>(mCarlo_dev, n, nThreads, nBlocks,
                                        state_x, state_y, offset);
        cudaDeviceSynchronize();
        cudaFree(state_x);
        cudaFree(state_y);
    }
    int mCarlo;
    cudaMemcpy(&mCarlo, mCarlo_dev, sizeof(int), cudaMemcpyDeviceToHost);
    double pi = (double) mCarlo / (double) n * 4;
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Pi = %.15f\n", pi);
    printf("Diff = %.15f\n", abs(M_PI - pi));
    printf("Time: ");
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf(" ms\n");
    cudaFree(mCarlo_dev);
    return 0;
}
