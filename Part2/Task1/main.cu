
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>


int
main(int argc, char **argv)
{

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("Устройства CUDA недоступны\n");
    }
    else
    {
        printf("Найдено(ы) %d CUDA-устроство(устройства/устройств):\n", deviceCount);
    }

    int dev;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Устройство %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Общий объем графической памяти:                            %.0f Мегабайт (%llu байт)\n",
               (float) deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("  Объём разделяемой памяти в пределах блока:                 %lu байт\n",
               deviceProp.sharedMemPerBlock);
        printf("  Число регистров в пределах блока:                          %d\n",
               deviceProp.regsPerBlock);
        printf("  Размер варпа:                                              %d\n",
               deviceProp.warpSize);
        printf("  Максимально допустимое число потоков в блоке:              %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Версия вычислительных возможностей:                        %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Число мультипроцессоров:                                   %d\n",
               deviceProp.multiProcessorCount);
        printf("  Тактовая частота ядра:                                     %.0f МГц (%0.2f ГГц)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("  Объём кэша 2 уровня:                                       %d байт\n",
               deviceProp.l2CacheSize);
        printf("  Ширина шины памяти:                                        %d бит\n",
               deviceProp.memoryBusWidth);
        printf("  Максимальная размерность при конфигурации потоков в блоке: %dx%dx%d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Максимальная размерность при конфигурации блоков в сетке:  %dx%dx%d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Максимальный объём константной памяти:                     %lu байт\n",
               deviceProp.totalConstMem);
        int peakClock = 0;
        cudaDeviceGetAttribute(&peakClock, cudaDevAttrClockRate, dev);
        printf("  Пиковая частота карты:                                     %.0f Мгц (%.2f ГГц)\n",
                peakClock * 1e-3f, peakClock * 1e-6f);
        printf("  *Конец описания устройства %d*\n", dev);
    }
    exit(EXIT_SUCCESS);
}
