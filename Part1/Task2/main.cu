#include <stdio.h>
#include <iostream>
// ядро
__global__ void add( int *a, int *b, int *c ) {
    *c = *a + *b;
}
//главная функция
int main()
{
    // переменные на CPU
    int a, b, c;
    printf("Введите а:\n");
    std::cin >> a;
    printf("Введите b:\n");
    std::cin >> b;
    // переменные на GPU
    int *dev_a, *dev_b, *dev_c;
    int size = sizeof( int );
    //размерность
    // выделяем память на GPU
    cudaMalloc(
            (void**)&dev_a, size );
    cudaMalloc(
            (void**)&dev_b, size );
    cudaMalloc(
            (void**)&dev_c, size );
// инициализация переменных
// копирование информации с CPU на GPU
    cudaMemcpy( dev_a, &a, size, cudaMemcpyHostToDevice
    );
    cudaMemcpy( dev_b, &b, size, cudaMemcpyHostToDevice
    );
// вызов ядра
    add<<< 1, 1 >>>( dev_a, dev_b, dev_c );
// копирование результата работы ядра с GPU на CPU
    cudaMemcpy( &c, dev_c, size, cudaMemcpyDeviceToHost
    );
// вывод информации
    printf("%d + %d = %d\n", a, b, c);
// очищение памяти на GPU
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    return 0;
}
