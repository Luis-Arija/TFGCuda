#include <stdio.h>
#define N 10000000
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

__global__ void vector_add(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void multiplication(float* A, float* B, float* C, int M) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    cuda_hello<<<1,1>>>(); 
}