#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;

__global__ void log_float(float* d_arr, int N, int M) { //ln x для float
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    // int ind = threadIdx.x;

    if (ind < N * M) {
        d_arr[ind] = __logf(ind + 0.01);
    }
}

__global__ void log_double(double* d_arr, int N, int M) { //ln x для double
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < N * M) {
        d_arr[ind] = logf(ind + 0.01);
    }
}


int main()
{
    // FLOAT
    int N = 256;
    int M;
    for(M = 1; M <= 256; M *= 2){
        float arrLog[N * M];
        float ans_arrLog[N * M];
        cout << N * M << ",";
        float * d_arr;
        for (int i = 0; i < N * M; i++) {
            arrLog[i] = logf(i + 0.01);
        }
        cudaMalloc((void**)&d_arr, N * M * sizeof(float));
        // Инициализация события
        cudaEvent_t start_1, stop_1;
        float elapsedTime;
        // Создаем события
        cudaEventCreate(&start_1);
        cudaEventCreate(&stop_1);
        // Запись события
        cudaEventRecord(start_1, 0);
        log_float <<<M, N >>> (d_arr, N, M);
        cudaMemcpy(&ans_arrLog, d_arr, N * M *sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_1, 0);
        // Ожидание завершения работы ядра
        cudaEventSynchronize(stop_1);
        cudaEventElapsedTime(&elapsedTime, start_1, stop_1);
        // Уничтожение событий
        cudaEventDestroy(start_1);
        cudaEventDestroy(stop_1);
        float errLog = 0.0;
        float errExpf = 0.0;
        for (int i = 0; i < N * M; i++) {
            errLog += abs(arrLog[i] - ans_arrLog[i])/ arrLog[i];
        }
        cudaFree(d_arr);
        printf("%.15f,", errLog);
        printf("%f\n", elapsedTime);
    }

    printf("------------------------------\n");

    // DOUBLE

    for(M = 1; M <= 256; M *= 2){
        cout << N * M << ",";
        double arrLog_d[N * M];
        double ans_arrLog_d[N * M];
        double* d_arr_d;
        for (int i = 0; i < N * M; i++) {
            arrLog_d[i] = log(i + 0.01);
        }
        cudaMalloc((void**)&d_arr_d, N * M * sizeof(double));
        // Инициализация события
        cudaEvent_t start_2, stop_2;
        float elapsedTime_2;
        // Создаем события
        cudaEventCreate(&start_2);
        cudaEventCreate(&stop_2);
        // Запись события
        cudaEventRecord(start_2, 0);
        log_double <<<M, N >>> (d_arr_d, N, M);
        cudaMemcpy(&ans_arrLog_d, d_arr_d, N * M * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_2, 0);
        // Ожидание завершения работы ядра
        cudaEventSynchronize(stop_2);
        cudaEventElapsedTime(&elapsedTime_2, start_2, stop_2);
        // Уничтожение событий
        cudaEventDestroy(start_2);
        cudaEventDestroy(stop_2);
        double errLog_d = 0.0;
        double errExpf_d = 0.0;
        for (int i = 0; i < N * M; i++) {
            errLog_d += abs(arrLog_d[i] - ans_arrLog_d[i])/ arrLog_d[i];
        }
        printf("%.15f,", errLog_d);
        printf("%f\n", elapsedTime_2);
    }
    return 0;
}