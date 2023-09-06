#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void HelloWorld()
{
    printf("Hello world, %d, %d\n", blockIdx.x,
           threadIdx.x);
}
int main()
{
    HelloWorld <<<2, 5>>>();
// хост ожидает завершения работы девайса
    cudaDeviceSynchronize();
// ожидаем нажатия любой клавиши
    getchar();
    return 0;
}

