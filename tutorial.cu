#include <stdio.h>
#include <cuda.h>
#define N 10

__global__ void AddInts(int *out, int *a, int *b) {
	out[0] = a[0] + b[0];
}

__global__ void AddVector(int *out, int *a, int *b) { 
	int nThread = threadIdx.x;
	out[nThread] = a[nThread] + b[nThread];
	printf("Thread id is: %d \n", nThread);
}

__global__ void AddMatrix(int** out, int** a, int** b) {
	int nThread = threadIdx.x;
	int nBlock = blockIdx.x;
	out[nBlock][nThread] = a[nBlock][nThread] + b[nBlock][nThread];
	}

__global__ void MulMatrix(int** out, int** a, int** b) {
	int nBlock = blockIdx.x; //Fila
	int nThread = threadIdx.x; //Columna
	int sumatorio = 0;

	for (int i = 0; i < N; i++) {
		sumatorio += a[nBlock][i] * b[i][nThread];
	}
	out[nBlock][nThread] = sumatorio;
}


int main() {
	
	//Suma simple
	int a = 8; 
	int b = 9;
	int out = 0;
	int *d_a, *d_b, *d_out;

	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));
	cudaMalloc(&d_out, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, &out, sizeof(int), cudaMemcpyHostToDevice);

	AddInts << <1, 1 >> > (d_out, d_a, d_b);

	cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("The answer is %d\n", out);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	//Suma de Vectores
	//int A[2] = { 1, 2 };
	//int B[2] = { 3, 4 };
	//int OUT[2] = { 0, 0 };

	int* a2;
	int* b2;
	int* out2;
	int* d_a2, * d_b2, * d_out2;
	// Crear vectores
	a2 = (int*)malloc(sizeof(int) * N);
	b2 = (int*)malloc(sizeof(int) * N);
	out2 = (int*)malloc(sizeof(int) * N);

	for (int i = 0; i < N; i++) {
		a2[i] = i+1;
		b2[i] = i+N+1;
		out2[i] = 0;
	}

	cudaMalloc((void**)&d_a2, sizeof(int) * N);
	cudaMalloc((void**)&d_b2, sizeof(int) * N);
	cudaMalloc((void**)&d_out2, sizeof(int) * N);

	cudaMemcpy(d_a2, a2, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b2, b2, sizeof(int) * N, cudaMemcpyHostToDevice);

	AddVector << <1, N >> > (d_out2, d_a2, d_b2);

	cudaMemcpy(out2, d_out2, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a2[i], b2[i], out2[i]);
	}

	

	return 0;
}