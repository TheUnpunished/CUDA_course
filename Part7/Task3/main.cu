#include <iostream>
#include <ctime>

#define BLOCK_SIZE 32
#define GRID_SIZE 16
#define BASE_TYPE int
#define n 100000

__global__ void scalar (BASE_TYPE* a, BASE_TYPE* sum, int offset){
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idx += offset * BLOCK_SIZE * GRID_SIZE;
    __shared__ BASE_TYPE a_sh[BLOCK_SIZE];
    if(idx < n)
        a_sh[threadIdx.x] = a[idx];
    else
        a_sh[threadIdx.x] = 0.0;
    __syncthreads();
    if(threadIdx.x == 0){
        BASE_TYPE sum_temp = 0.0;
        for(int i = 0; i < BLOCK_SIZE; i ++)
            sum_temp += a_sh[i] * a_sh[i];
        sum[blockIdx.x] = sum_temp;
    }
}

int main() {
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);
    std::srand(time(nullptr));
    size_t size = sizeof(BASE_TYPE) * n;
    size_t size_sum = sizeof(BASE_TYPE) * GRID_SIZE;
    BASE_TYPE* a = (BASE_TYPE*) malloc(size);
    for(int i = 0; i < n; i ++){
        a[i] = std::rand() % 201 - 100;
    }
    BASE_TYPE* a_dev;
    cudaMalloc((void **) &a_dev, size);
    cudaMemcpy(a_dev, a, size, cudaMemcpyHostToDevice);
    BASE_TYPE* sum_dev;
    cudaMalloc((void **) &sum_dev, size_sum);
    int offset_max;
    if((n) % (BLOCK_SIZE * GRID_SIZE) != 0)
        offset_max = n / (BLOCK_SIZE * GRID_SIZE) + 1;
    else
        offset_max = n / (BLOCK_SIZE * GRID_SIZE);
    BASE_TYPE sum = 0.0;
    BASE_TYPE* sum_h = (BASE_TYPE*) malloc(size_sum);
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int offset = 0; offset <= offset_max; offset ++){
        scalar<<<dimGrid, dimBlock>>>(a_dev, sum_dev, offset);
        cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for(int i = 0; i < GRID_SIZE; i ++){
            sum += sum_h[i];
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    free(a);
    free(sum_h);
    cudaFree(a_dev);
    cudaFree(sum_dev);
    std::cout << sum << std::endl;
    double sum_sqrt = sqrt(sum);
    std::cout << sum_sqrt << std::endl;
    printf("Время работы на ГПУ: %.2f мс\n", elapsedTime);
    return 0;
}
