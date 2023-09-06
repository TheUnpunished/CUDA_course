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

__global__ void integral_rect(BASE_TYPE* sum, double step, int offset){
    __shared__ BASE_TYPE sum_shared[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idx += offset * BLOCK_SIZE * GRID_SIZE;
    BASE_TYPE x = x_low + step * idx;
    if(idx < n)
        sum_shared[threadIdx.x] = fx(x) * step;
    else
        sum_shared[threadIdx.x] = 0.0;
    __syncthreads();
    if(threadIdx.x == 0){
        BASE_TYPE sum_temp = 0;
        for(int i = 0; i < BLOCK_SIZE; i ++){
            sum_temp += sum_shared[i];
        }
        sum[blockIdx.x] = sum_temp;
    }
}

__global__ void integral_trap(BASE_TYPE* sum, double step, int offset){
    __shared__ BASE_TYPE sum_shared[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idx += offset * BLOCK_SIZE * GRID_SIZE;
    BASE_TYPE x = x_low + step * idx;
    if(idx < n)
        sum_shared[threadIdx.x] = (fx(x) + fx(x + step)) / 2 * step;
    else
        sum_shared[threadIdx.x] = 0.0;
    __syncthreads();
    if(threadIdx.x == 0){
        BASE_TYPE sum_temp = 0;
        for(int i = 0; i < BLOCK_SIZE; i ++){
            sum_temp += sum_shared[i];
        }
        sum[blockIdx.x] = sum_temp;
    }
}


__global__ void integral_simpson(BASE_TYPE* sum, double step, int offset){
    __shared__ BASE_TYPE sum_shared[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idx += offset * BLOCK_SIZE * GRID_SIZE;
    BASE_TYPE x = x_low + step * idx;
    if(idx < n)
        sum_shared[threadIdx.x] = step / 6 * (fx(x) + 4 * fx((2 * x + step) /  2) + fx(x + step));
    else
        sum_shared[threadIdx.x] = 0.0;
    __syncthreads();
    if(threadIdx.x == 0){
        BASE_TYPE sum_temp = 0;
        for(int i = 0; i < BLOCK_SIZE; i ++){
            sum_temp += sum_shared[i];
        }
        sum[blockIdx.x] = sum_temp;
    }
}

int main() {
    dim3 dimGrid(GRID_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    double step = abs((x_high - x_low)) / n;
    size_t size_sum = sizeof(BASE_TYPE) * BLOCK_SIZE;
    int offset_max;
    if((n) % (BLOCK_SIZE * GRID_SIZE) != 0)
        offset_max = n / (BLOCK_SIZE * GRID_SIZE) + 1;
    else
        offset_max = n / (BLOCK_SIZE * GRID_SIZE);
    BASE_TYPE* sum_dev;
    cudaMalloc((void **) &sum_dev, size_sum);
    BASE_TYPE sum = 0.0;
    BASE_TYPE* sum_h = (BASE_TYPE*) malloc(size_sum);
    for(int offset = 0; offset <= offset_max; offset ++){
        integral_rect<<<dimGrid, dimBlock>>>(sum_dev, step, offset);
        cudaDeviceSynchronize();
        cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
        for(int i = 0; i < GRID_SIZE; i ++){
            sum += sum_h[i];
        }
    }
    BASE_TYPE sum_rect = sum;
    sum = 0.0;
    for(int offset = 0; offset <= offset_max; offset ++){
        integral_trap<<<dimGrid, dimBlock>>>(sum_dev, step, offset);
        cudaDeviceSynchronize();
        cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
        for(int i = 0; i < GRID_SIZE; i ++){
            sum += sum_h[i];
        }
    }
    BASE_TYPE sum_trap = sum;
    sum = 0.0;
    for(int offset = 0; offset <= offset_max; offset ++){
        integral_simpson<<<dimGrid, dimBlock>>>(sum_dev, step, offset);
        cudaDeviceSynchronize();
        cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
        for(int i = 0; i < GRID_SIZE; i ++){
            sum += sum_h[i];
        }
    }
    free(sum_h);
    cudaFree(sum_dev);
    std::cout << sum_rect << std::endl;
    std::cout << sum_trap << std::endl;
    std::cout << sum << std::endl;
    return 0;
}
