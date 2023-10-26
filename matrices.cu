#include <stdio.h>
#include <cuda.h>
#define N 16
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))


//Funciones Kernel
__global__ void AddInts(int* out, int* a, int* b) {
	out[0] = a[0] + b[0];
}

__global__ void AddVector(int* out, int* a, int* b) {
	int nThread = threadIdx.x;
	out[nThread] = a[nThread] + b[nThread];
}

//Es una versi�n m�s completa de AddVector
__global__ void AddMatrix(int* out, int* a, int* b) {
	int nThread = threadIdx.x;
	int nBlock = blockIdx.x;
	int blockDimension = blockDim.x;
	int id = nBlock * blockDimension + nThread;
	out[id] = a[id] + b[id];
}

__global__ void MulMatrix(int* out, int* a, int* b, int nColumns1) {
	int nBlock = blockIdx.x; //Fila
	int nThread = threadIdx.x; //Columna
	int blockSize = blockDim.x; //nColumnas final
	int sumatorio = 0;
	int id = nBlock * blockSize + nThread;

	for (int i = 0; i < nColumns1; i++) {
		sumatorio += a[nBlock * nColumns1 + i] * b[i*blockSize + nThread];
	}
	out[id] = sumatorio;
}

__global__ void MoveMatrix(int* to, int* from) {
	int nBlock = blockIdx.x; //Fila
	int nThread = threadIdx.x; //Columna
	int blockSize = blockDim.x; //nColumnas final
	int id = nBlock * blockSize + nThread;
	to[id] = from[id];
}



//Funciones normales
void SumaSimple(int a, int b) {
	int out = 0;
	int* d_a, * d_b, * d_out;

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
}

void SumaVectorial(int vector1[], int vector2[], int size1, int size2, bool wantPrint) {
	
	//Asegurarse de que el tama�o es el mismo
	/*Intento de calcular el tama�o dentro de la funcion: IMPOSIBLE
	size_t size1 = sizeof(vector1) / sizeof(vector1[0]);
	size_t size2 = sizeof(vector2) / sizeof(vector2[0]);
	printf("Size of vector1: %zd\n", size1);
	printf("Size of vector2: %zd\n", size2);
	*/
	if (size1 == size2) {
		//Declarar variables
		int* vOut = (int*)malloc(sizeof(int) * size1);
		int* d_v1, * d_v2, * d_vOut;

		//Hacer espacio en el GPU
		cudaMalloc((void**)&d_v1, sizeof(int) * size1);
		cudaMalloc((void**)&d_v2, sizeof(int) * size1);
		cudaMalloc((void**)&d_vOut, sizeof(int) * size1);

		//Pasar informaci�n al GPU
		cudaMemcpy(d_v1, vector1, sizeof(int) * size1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, vector2, sizeof(int) * size1, cudaMemcpyHostToDevice);

		//Invocar funcion de suma
		AddVector << <1, size1>> > (d_vOut, d_v1, d_v2);

		//Pasar informaci�n resultante de vuelta
		cudaMemcpy(vOut, d_vOut, sizeof(int) * size1, cudaMemcpyDeviceToHost);

		if (wantPrint) {
			//Print resultado
			for (int i = 0; i < size1; i++) {
				printf("%d + %d = %d\n", vector1[i], vector2[i], vOut[i]);
			}
		}
		

		//liberar el espacio usado en el GPU
		cudaFree(d_v1);
		cudaFree(d_v2);
		cudaFree(d_vOut);
	}
	else {
		printf("Mismatch in vector sizes");
	}


}

//Como no hay manera de averiguar el numero de filas y columnas, si los tama�os cuadran trabaja el resultado en funci�n de nrowsxncolumns
void SumaMatricial(int matriz1[], int matriz2[], int size1, int size2, int nRows, int nColumns, bool wantPrint) {

	//Asegurarse de que el tama�o es el mismo es imposible dentro de la funci�n
	/*
	int size1 = sizeof(matriz1) / sizeof(int);
	int size2 = sizeof(matriz2) / sizeof(int);
	printf("Size: %d    ",expectedSize);
	*/
	int expectedSize = nRows * nColumns;
	

	if (size1 == expectedSize && size1 == size2) {
		//Declarar variables
		int* vOut = (int*)malloc(sizeof(int) * expectedSize);
		int* d_v1, * d_v2, * d_vOut;

		//Hacer espacio en el GPU
		cudaMalloc((void**)&d_v1, sizeof(int) * expectedSize);
		cudaMalloc((void**)&d_v2, sizeof(int) * expectedSize);
		cudaMalloc((void**)&d_vOut, sizeof(int) * expectedSize);

		//Pasar informaci�n al GPU
		cudaMemcpy(d_v1, matriz1, sizeof(int) * expectedSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, matriz2, sizeof(int) * expectedSize, cudaMemcpyHostToDevice);

		//Invocar funcion de suma
		AddMatrix << <nRows, nColumns >> > (d_vOut, d_v1, d_v2);

		//Pasar informaci�n resultante de vuelta
		cudaMemcpy(vOut, d_vOut, sizeof(int) * expectedSize, cudaMemcpyDeviceToHost);

		//Print resultado
		if (wantPrint) {
			printf("\n%d rows, %d columns. Output:\n", nRows, nColumns);
			for (int i = 0; i < nRows; i++) {
				for (int j = 0; j < nColumns; j++) {
					printf("%d     ", vOut[i * nColumns + j]);
				}
				printf("\n");
			}
		}
	}
	else {
		printf("Mismatch in array sizes");
	}
}

//
void MultiplicacionMatricial(int matriz1[], int matriz2[], int size1, int size2, int nRows1, int nColumns1, int nRows2, int nColumns2, bool wantPrint) {
	//Comprobar que los tama�os cuadran
	if (size1 == nRows1 * nColumns1 && size2 == nRows2 * nColumns2) {
		//Comprobar que n1==m2
		if (nColumns1 == nRows2) {
			int expectedSizeOut = nRows1 * nColumns2;//m1xn1*m2xn2=m1xn2
			//Declarar variables
			int* vOut = (int*)malloc(sizeof(int) * expectedSizeOut);

			int* d_v1, * d_v2, * d_vOut;

			//Hacer espacio en el GPU
			cudaMalloc((void**)&d_v1, sizeof(int) * size1);
			cudaMalloc((void**)&d_v2, sizeof(int) * size2);
			cudaMalloc((void**)&d_vOut, sizeof(int) * expectedSizeOut);

			//Pasar informaci�n al GPU
			cudaMemcpy(d_v1, matriz1, sizeof(int) * size1, cudaMemcpyHostToDevice);
			cudaMemcpy(d_v2, matriz2, sizeof(int) * size2, cudaMemcpyHostToDevice);

			//Invocar funcion de multiplicacion
			MulMatrix << <nRows1, nColumns2 >> > (d_vOut, d_v1, d_v2, nColumns1);

			//Pasar informaci�n resultante de vuelta
			cudaMemcpy(vOut, d_vOut, sizeof(int) * expectedSizeOut, cudaMemcpyDeviceToHost);

			//Print resultado
			if (wantPrint) {
				printf("\nMatrix 1: %d rows, %d columns. Output:\n", nRows1, nColumns1);
				for (int i = 0; i < nRows1; i++) {
					for (int j = 0; j < nColumns1; j++) {
						printf("%d     ", matriz1[i * nColumns1 + j]);
					}
					printf("\n");
				}

				printf("\nMatrix 2: %d rows, %d columns. Output:\n", nRows2, nColumns2);
				for (int i = 0; i < nRows2; i++) {
					for (int j = 0; j < nColumns2; j++) {
						printf("%d     ", matriz2[i * nColumns2 + j]);
					}
					printf("\n");
				}

				printf("\n%d rows, %d columns. Output:\n", nRows1, nColumns2);
				for (int i = 0; i < nRows1; i++) {
					for (int j = 0; j < nColumns2; j++) {
						printf("%d     ", vOut[i * nColumns2 + j]);
					}
					printf("\n");
				}
			}
		}
		else {
			printf("Matriz 1 tiene un numero de columnas distinto al numero de filas de Matriz 2");
		}
	}
	else {
		printf("Los tama�os de las matrices no cuadran con las filas y columnas especificadas");
	}
}

void PotenciaMatricial(int matriz[], int size, int nRows, int nColumns, int nPotencia, bool wantPrint) {
	//Comprobar tama�o
	if (size == nRows * nColumns && nRows==nColumns) {
		//Declarar variables
		int* vOut = (int*)malloc(sizeof(int) * size);
		int* d_v1, * d_v2, * d_vOut;

		//Hacer espacio en el GPU
		cudaMalloc((void**)&d_v1, sizeof(int) * size);
		cudaMalloc((void**)&d_v2, sizeof(int) * size);
		cudaMalloc((void**)&d_vOut, sizeof(int) * size);

		//Pasar informaci�n al GPU
		cudaMemcpy(d_v1, matriz, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, matriz, sizeof(int) * size, cudaMemcpyHostToDevice);


		//For
		for (int i = 1; i < nPotencia; i++) {
			MulMatrix << <nRows, nColumns >> > (d_vOut, d_v1, d_v2, nColumns);
			MoveMatrix << <nRows, nColumns >> > (d_v1, d_vOut);
		}

		//Pasar informaci�n resultante de vuelta
		cudaMemcpy(vOut, d_vOut, sizeof(int) * size, cudaMemcpyDeviceToHost);

		//Print resultado
		if (wantPrint) {

			printf("\nOriginal Matrix : %d rows, %d columns. Output:\n", nRows, nColumns);
			for (int i = 0; i < nRows; i++) {
				for (int j = 0; j < nColumns; j++) {
					printf("%d     ", matriz[i * nColumns + j]);
				}
				printf("\n");
			}

			printf("\nMatrix : %d rows, %d columns. Output:\n", nRows, nColumns);
			for (int i = 0; i < nRows; i++) {
				for (int j = 0; j < nColumns; j++) {
					printf("%d     ", vOut[i * nColumns + j]);
				}
				printf("\n");
			}
		}
	}
	else {
		printf("Tama�o de Matriz incorrecto");
	}
}

int main() {

	//SumaSimple(3, 4);

	//Suma vectorial

	int vector1[N];
	int vector2[N];

	//size_t size1 = sizeof(vector1) / sizeof(vector1[1]);
	//size_t size2 = sizeof(vector2) / sizeof(vector2[1]);
	//printf("Size of vector1: %zd\n", size1);
	//printf("Size of vector2: %zd\n", size2);
	for (int i = 0; i < N; i++) {
		vector1[i] = i + 1;
		vector2[i] = i + N + 1;
	}
	SumaVectorial(vector1, vector2, NELEMS(vector1), NELEMS(vector2), false);
	SumaMatricial(vector1, vector2, NELEMS(vector1), NELEMS(vector2), 4, 4, false);
	MultiplicacionMatricial(vector1, vector2, NELEMS(vector1), NELEMS(vector2), 4, 4, 4, 4, false);

	PotenciaMatricial(vector1, NELEMS(vector1), 4, 4, 2, true);


	return 0;
}