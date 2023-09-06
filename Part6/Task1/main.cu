#include <iostream>
#include <ctime>


__global__ void matr_Mult(int* a, int* res, int xThreads, int yThreads, int nBlocks, int n_val){
    // idx + idy
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    for(int i = idx; i < n_val; i += nBlocks * xThreads){
        for(int j = idy; j < n_val; j += nBlocks * yThreads){
            for(int k = 0; k < n_val; k ++){
                // i строка * j столбец
//                res[i * n_val + j] += a[i * n_val + k] * a[k * n_val + j];
                // i строка * j строка
                res[i * n_val + j] += a[i * n_val + k] * a[j * n_val + k];
            }
        }
    }
}

#define x_threads 16
#define y_threads 16
#define n_blocks 16
#define n 100

int main() {
    dim3 dim_Block(x_threads, y_threads, 1);
    dim3 dim_Grid(n_blocks, n_blocks, 1);
    size_t matr_size = sizeof(int) * n * n;
    int *a = (int*) malloc(matr_size);
    srand(time(nullptr));
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++){
            if(i == j)
//                a[i * n + j] = std::rand() % 201 - 100;
                a[i * n + j] = 1;
            else
                a[i * n + j] = 0;
        }
    }
    int* a_dev;
    cudaMalloc((void **) &a_dev, matr_size);
    int* res_dev;
    cudaMalloc((void **) &res_dev, matr_size);
    cudaMemset(res_dev, 0, matr_size);
    cudaMemcpy(a_dev, a, matr_size, cudaMemcpyHostToDevice);
    matr_Mult<<<dim_Grid, dim_Block>>>(a_dev, res_dev, x_threads, y_threads, n_blocks, n);
    cudaDeviceSynchronize();
    cudaFree(a_dev);
    cudaMemcpy(a, res_dev, matr_size, cudaMemcpyDeviceToHost);
    bool ortogonal = true;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i != j){
                ortogonal = (a[i * n + j] == 0) && ortogonal;
            }
            else{
                ortogonal = (a[i * n + j] == 1) && ortogonal;
            }
        }
    }
    free(a);
    std::cout << ortogonal << std::endl;
    return 0;
}
