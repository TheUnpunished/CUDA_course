#include <iostream>

#define BLOCK_SIZE 32
#define GRID_SIZE 16
#define BASE_TYPE int
#define n 10000

texture<BASE_TYPE, 1, cudaReadModeElementType> texRef_a;
texture<BASE_TYPE, 1, cudaReadModeElementType> texRef_b;

__global__ void scalar(BASE_TYPE* sum){
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(int i = idx; i < n; i += BLOCK_SIZE * GRID_SIZE){
        sum[idx] += tex1Dfetch(texRef_a, i) * tex1D(texRef_b, i);
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
    for(int i = 0; i < n; i ++){
        a[i] = std::rand() % 201 - 100;
        b[i] = std::rand() % 201 - 100;
    }
    BASE_TYPE* a_dev;
    cudaMalloc((void **) &a_dev, size);
    cudaMemcpy(a_dev, a, size, cudaMemcpyHostToDevice);
    cudaBindTexture(0, texRef_a, a_dev, size);
    texRef_a.normalized = false;
    texRef_a.filterMode = cudaFilterModePoint;
    cudaArray* cuArray_b;
    cudaMallocArray(&cuArray_b, &texRef_b.channelDesc, n, 1);
    cudaMemcpyToArray(cuArray_b, 0, 0, b,
                      size, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(texRef_b, cuArray_b);
    texRef_b.normalized = false;
    texRef_b.filterMode = cudaFilterModePoint;
    BASE_TYPE* sum_dev;
    cudaMalloc((void **) &sum_dev, size_sum);
    cudaMemset(sum_dev, 0, size_sum);
    BASE_TYPE sum = 0.0;
    BASE_TYPE* sum_h = (BASE_TYPE*) malloc(size_sum);
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    scalar<<<dimGrid, dimBlock>>>(sum_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(sum_h, sum_dev, size_sum, cudaMemcpyDeviceToHost);
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
    cudaFree(a_dev);
    cudaFree(sum_dev);
    std::cout << sum << std::endl;
    std::cout << sum_check << std::endl;
    printf("Время работы на ГПУ: %.2f мс\n", elapsedTime);
    return 0;
}
