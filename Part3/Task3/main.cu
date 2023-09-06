#include <iostream>
#include <ctime>
#include "chrono"



int main() {
    unsigned char *data_local = NULL;
    int iterations = 100;
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t memSize = deviceProp.totalGlobalMem / 2;
    data_local = (unsigned char *) malloc(memSize);
    for(unsigned int i = 0; i < memSize / sizeof(unsigned char); i++){
        data_local[i] = (unsigned char)(i & 0xff);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    unsigned char *data_device;
    cudaMalloc((void **) &data_device, memSize);
    cudaEventRecord(start, 0);
    for(int i = 0; i < iterations; i ++){
        cudaMemcpy(data_device, data_local, memSize, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop, 0);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Время: %f\n", elapsedTime);
    float bandWidth = ((float)(1024) * memSize * (float)iterations) /
                      (elapsedTime * (float)(1 << 30));
    printf("Проп. способность: %f ГБ/с", bandWidth);
    cudaFree(data_device);
    free(data_local);
    return 0;
}
