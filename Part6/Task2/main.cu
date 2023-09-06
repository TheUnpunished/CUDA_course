#include <iostream>

__global__ void matr_Mult(int* a, int* b, int* res, int xThreads, int yThreads, int nBlocks, int n_val){
    // idx + idy
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    for(int i = idx; i < n_val; i += nBlocks * xThreads){
        for(int j = idy; j < n_val; j += nBlocks * yThreads){
            for(int k = 0; k < n_val; k ++){
                // i строка * j столбец
                res[i * n_val + j] += a[i * n_val + k] * b[k * n_val + j];
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
            a[i * n + j] = std::rand() % 201 - 100;
        }
    }
    int *b = (int*) malloc(matr_size);
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++){
            if(i == j)
                b[i * n + j] = 1;
            else
                b[i * n + j] = 0;
        }
    }
    int* a_dev;
    cudaMalloc((void **) &a_dev, matr_size);
    int* b_dev;
    cudaMalloc((void **) &b_dev, matr_size);
    int* res_dev;
    cudaMalloc((void **) &res_dev, matr_size);
    cudaMemset(res_dev, 0, matr_size);
    cudaMemcpy(a_dev, a, matr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, matr_size, cudaMemcpyHostToDevice);
    matr_Mult<<<dim_Grid, dim_Block>>>(a_dev, b_dev, res_dev, x_threads, y_threads, n_blocks, n);
    cudaDeviceSynchronize();
    int* ab = (int*) malloc(matr_size);
    cudaMemcpy(ab, res_dev, matr_size, cudaMemcpyDeviceToHost);
    cudaMemset(res_dev, 0, matr_size);
    matr_Mult<<<dim_Grid, dim_Block>>>(b_dev, a_dev, res_dev, x_threads, y_threads, n_blocks, n);
    cudaDeviceSynchronize();
    int* ba = (int*) malloc(matr_size);
    cudaMemcpy(ba, res_dev, matr_size, cudaMemcpyDeviceToHost);
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(res_dev);
    free(a);
    free(b);
    bool commutating = true;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            commutating = (ab[i * n + j] == ba[i * n + j]) && commutating;
        }
    }
    std::cout << commutating << std::endl;
    return 0;
}
