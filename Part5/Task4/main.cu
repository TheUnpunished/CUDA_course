#include <iostream>

__global__ void scalar(float *x, float *y, float *sum, int n, int nThreads, int nBlocks){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float m;
    for (int i = idx; i < n; i += nThreads * nBlocks){
        m = x[i] * y[i];
        sum[idx] += m;
    }
}

__global__ void decr(float *a, float *b, float *result, int n, int nThreads, int nBlocks){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = idx; i < n; i += nThreads * nBlocks){
        float res = a[i] - b[i];
        result[i] = res;
    }
}

__global__ void mult(float *a, float b, float *result, int n, int nThreads, int nBlocks){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = idx; i < n; i += nThreads * nBlocks){
        float res = a[i] * b;
        result[i] = res;
    }
}


int main() {
    int n = 100;
    int nThreads = 512;
    int nBlocks = 100;
    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1,1);
    size_t size = sizeof(float) * n;
    float a[n][n];
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++)
            if(j >= i)
                a[i][j] = 1.0;
            else
                a[i][j] = 0.0;
    }
    float** b = new float*[n];
    for(int i = 0; i < n; i++){
        b[i] = new float[n];
    }
    for(int i = 0; i < n; i++){
        b[0][i] = a[0][i];
    }
    float *ai;
    cudaMalloc((void **) &ai, size);
    float *bk;
    cudaMalloc((void **) &bk, size);
    float *dev_temp;
    cudaMalloc((void **) &dev_temp, size);
    float *tempLocal;
    tempLocal = (float*) malloc(size);
    for (int i = 1; i < n; i ++){
        float sum[n];
        for(int k = 0; k < n; k++){
            sum[k] = 0;
        }
        for(int k = 0; k < n; k++){
            b[i][k] = 0;
        }
        cudaMemcpy(ai, a[i], size, cudaMemcpyHostToDevice);
        for(int k = i - 1; k >= 0; k --){
            float scalar_1 = 0;
            float scalar_2 = 0;
            cudaMemcpy(bk, b[k], size, cudaMemcpyHostToDevice);
            cudaMemset(dev_temp, 0.0, size);
            scalar<<<dimGrid, dimBlock>>>(ai, bk, dev_temp, n, nThreads, nBlocks);
            cudaDeviceSynchronize();
            cudaMemcpy(tempLocal, dev_temp, size, cudaMemcpyDeviceToHost);
            for(int i = 0; i < n; i ++){
                scalar_1 += tempLocal[i];
            }
            cudaMemset(dev_temp, 0, size);
            scalar<<<dimGrid, dimBlock>>>(bk, bk, dev_temp, n, nThreads, nBlocks);
            cudaDeviceSynchronize();
            cudaMemcpy(tempLocal, dev_temp, size, cudaMemcpyDeviceToHost);
            for(int i = 0; i < n; i ++){
                scalar_2 += tempLocal[i];
            }
            float scalar_res = scalar_1 / scalar_2;
            cudaMemset(dev_temp, 0, size);
            mult<<<dimGrid, dimBlock>>>(bk, scalar_res, dev_temp, n, nThreads, nBlocks);
            cudaDeviceSynchronize();
            cudaMemcpy(tempLocal, dev_temp, size, cudaMemcpyDeviceToHost);
            for(int t = 0; t < n; t ++){
                sum[t] += tempLocal[t];
            }
        }
        for(int k = 0; k < n; k ++){
            b[i][k] = a[i][k] - sum[k];
        }
    }
    double sum = 0.0;
    for(int i = 0; i < n; i ++){
        for(int k = 0; k < n; k ++){
            if(k != i){
                cudaMemcpy(ai, b[i], size, cudaMemcpyHostToDevice);
                cudaMemcpy(bk, b[k], size, cudaMemcpyHostToDevice);
                cudaMemset(dev_temp, 0, size);
                scalar<<<dimGrid, dimBlock>>>(ai, bk, dev_temp, n, nThreads, nBlocks);
                cudaMemcpy(tempLocal, dev_temp, size, cudaMemcpyDeviceToHost);
                for(int j = 0; j < n; j ++){
                    sum += tempLocal[j];
                }
            }
        }
    }
    printf("%.5f\n", sum);
    return 0;
}
