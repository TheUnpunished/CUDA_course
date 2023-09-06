#include <iostream>

__device__ double fx(double x){
    return 5 * x + 10;
}

#define BLOCK_SIZE 32
#define GRID_SIZE 16
#define BASE_TYPE double
#define n 100000000
#define x_low 0
#define x_high 3.141592653589793238

__constant__ BASE_TYPE step_dev;

__global__ void integral_rect(BASE_TYPE* sum){
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(int i = idx; i < n; i += BLOCK_SIZE * GRID_SIZE){
        BASE_TYPE x = x_low + step_dev * i;
        sum[idx] += fx(x) * step_dev;
    }
}

int main() {
    dim3 dimGrid(GRID_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    BASE_TYPE step_h = abs((x_high - x_low)) / n;
    cudaMemcpyToSymbol(step_dev, &step_h, sizeof(BASE_TYPE), 0, cudaMemcpyHostToDevice);
    size_t size_sum = sizeof(BASE_TYPE) * BLOCK_SIZE * GRID_SIZE;
    BASE_TYPE* sum_dev;
    cudaMalloc((void **) &sum_dev, size_sum);
    cudaMemset(sum_dev, 0, size_sum);
    BASE_TYPE sum = 0.0;
    BASE_TYPE* sum_h = (BASE_TYPE*) malloc(size_sum);
    integral_rect<<<dimGrid, dimBlock>>>(sum_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
    for(int i = 0; i < GRID_SIZE * BLOCK_SIZE; i ++){
        sum += sum_h[i];
    }
    std::cout << sum << std::endl;
    free(sum_h);
    cudaFree(sum_dev);
    return 0;
}
