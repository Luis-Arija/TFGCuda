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

    int myNumbers[] = { 25, 50, 75, 100 };
    int i;

    for (i = 0; i < 4; i++) {
        printf("%d\n", myNumbers[i]);
    }

    float* a, * b, * out;

    // Allocate memory
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add << <1, 1 >> > (out, a, b, N);

    //multiplication << <1, 1 >> > ();
    return 0;
}