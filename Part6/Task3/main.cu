#include <iostream>


__global__ void matrAdd(int* a[], int* b[], int* res[], int xThreads, int yThreads, int nBlocks, int n_val){
    // idx + idy
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    for(int i = idx; i < n_val; i += nBlocks * xThreads){
        for(int j = idy; j < n_val; j += nBlocks * yThreads){
            res[i][j] += a[i][j] + b[i][j];
        }
    }
}

#define x_threads 16
#define y_threads 16
#define n_blocks 16
#define n 5

int main() {
    dim3 dim_Block(x_threads, y_threads, 1);
    dim3 dim_Grid(n_blocks, n_blocks, 1);
    size_t line_size = sizeof(int) * n;
    size_t pointer_size = sizeof(int *) * n;
    int a[n][n];
    srand(time(nullptr));
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++){
            a[i][j] = std::rand() % 201 - 100;
        }
    }
    int b[n][n];
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++){
            b[i][j] = a[i][j];
        }
    }
    int** a_dev_host = (int **) malloc(pointer_size);
    for(int i = 0; i < n; i ++){
        cudaMalloc((void **) &a_dev_host[i], line_size);
        cudaMemcpy(a_dev_host[i], &a[i][0], line_size, cudaMemcpyHostToDevice);
    }
    int ** a_dev_dev;
    cudaMalloc((void ***) &a_dev_dev, pointer_size);
    cudaMemcpy(a_dev_dev, a_dev_host, pointer_size, cudaMemcpyHostToDevice);
    int** b_dev_host = (int **) malloc(pointer_size);
    for(int i = 0; i < n; i ++){
        cudaMalloc((void **) &b_dev_host[i], line_size);
        cudaMemcpy(b_dev_host[i], &b[i][0], line_size, cudaMemcpyHostToDevice);
    }
    int ** b_dev_dev;
    cudaMalloc((void ***) &b_dev_dev, pointer_size);
    cudaMemcpy(b_dev_dev, b_dev_host, pointer_size, cudaMemcpyHostToDevice);
    int** res_dev_host = (int **) malloc(n * sizeof(int *));
    for(int i = 0; i < n; i ++){
        cudaMalloc((void **) &res_dev_host[i], line_size);
        cudaMemset(res_dev_host[i], 0, line_size);
    }
    int ** res_dev_dev;
    cudaMalloc((void ***) &res_dev_dev, pointer_size);
    cudaMemcpy(res_dev_dev, res_dev_host, pointer_size, cudaMemcpyHostToDevice);
    matrAdd<<<dim_Grid, dim_Block>>>(a_dev_dev, b_dev_dev, res_dev_dev, x_threads, y_threads, n_blocks, n);
    cudaDeviceSynchronize();
    int ab[n][n];
    for(int i = 0; i < n; i ++){
        cudaMemcpy(&ab[i][0], res_dev_host[i], line_size, cudaMemcpyDeviceToHost);
    }
    bool dbl = true;
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++){
            dbl = (ab[i][j] == (a[i][j] * 2)) && dbl;
        }
    }
    std::cout << dbl << std::endl;
    return 0;
}
