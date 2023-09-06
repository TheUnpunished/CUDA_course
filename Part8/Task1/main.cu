#include <iostream>

#define BLOCK_SIZE 32
#define GRID_SIZE 16
#define BASE_TYPE int
#define n 1000

__constant__ BASE_TYPE a_dev[n];
__constant__ BASE_TYPE b_dev[n];

__global__ void scalar (BASE_TYPE* sum){
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(int i = idx; i < n; i+= BLOCK_SIZE * GRID_SIZE) {
        sum[idx] += a_dev[i] * b_dev[i];
    }
}

int main() {
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);
    std::srand(time(nullptr));
    size_t size = sizeof(BASE_TYPE) * n;
    size_t size_sum = sizeof(BASE_TYPE) * GRID_SIZE * BLOCK_SIZE;
    BASE_TYPE* a = (BASE_TYPE*) malloc(size);
    BASE_TYPE* b = (BASE_TYPE*) malloc(size);
    BASE_TYPE* sum_dev;
    cudaMalloc((void **) &sum_dev, size_sum);
    for(int i = 0; i < n; i ++){
        a[i] = std::rand() % 201 - 100;
        b[i] = std::rand() % 201 - 100;
    }
    cudaMemcpyToSymbol (a_dev, a, size,0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol (b_dev, b, size,0, cudaMemcpyHostToDevice);
    cudaMemset(sum_dev, 0, size_sum);
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    BASE_TYPE sum = 0.0;
    BASE_TYPE* sum_h = (BASE_TYPE*) malloc(size_sum);
    scalar<<<dimGrid, dimBlock>>>(sum_dev);
    cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < GRID_SIZE * BLOCK_SIZE; i ++){
        sum += sum_h[i];
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    BASE_TYPE sum_check = 0.0;
    for(int i = 0; i < n; i ++)
        sum_check += a[i] * b[i];
    free(a);
    free(b);
    free(sum_h);
    cudaFree(sum_dev);
    std::cout << sum << std::endl;
    std::cout << sum_check << std::endl;
    printf("Время работы на ГПУ: %.2f мс\n", elapsedTime);
    return 0;
}
