#include <stdio.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <Windows.h>
#define N 16
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))


/*
Funciones Kernel : Las funciones que trabajan con datos dentro de la GPU.
Para invocar una función Kernel, se escribe lo siguiente: Nombre_Función << <NBloques, TamañoBloque >> > (Arg1, Arg2,..., ArgN);
Así se creará un número de threads paralelos igual a NBloques*TamañoBloque, y cada thread realizará la función Nombre_Funcion.   
Los argumentos a los que tienen acceso son en todo momento iguales, pero el resultado será distinto dependiendo del ID de dicho thread. 
El ID del thread se suele calcular al comienzo de la función. Este es igual al ID de bloque del thread (BlockId) por el tamaño de los bloques  más el ID del thread dentro del bloque
Matemáticamente: Id= BlockId*BlockSize+ThreadId; Su valor está dentro de {0, (NBloques*TamañoBloques)-1}

Como los Kernel tienen acceso a esos datos, es lógico crear funciones que trabajen en base a ellos, y en cierto modo dependan de ellos. 
Se explicará los requisitos de invocación para cada función. 
*/

//Add Ints es una simple suma. Se usa <<<1,1>>> porque hacerlo más veces no añade nada. 
__global__ void AddInts(int* out, int* a, int* b) {
	out[0] = a[0] + b[0];
}

//Add Vector es una suma de vectores representados como arrays. Usa <<<1, ArraySize>>>, y cada thread calcula el valor de Out en su id. 
__global__ void AddVector(int* out, int* a, int* b) {
	int nThread = threadIdx.x;
	out[nThread] = a[nThread] + b[nThread];
}

//AddMatrix es una versión más completa de AddVector. Suma dos matrices representadas como arrays. 
//Teóricamente lo que hace esta función lo puede hacer AddVector si el tamaño del vector está por debajo de 1029. (Nmax de threads es 1028)
//La función se invoca con <<<NFilas,Ncolumnas>>>
//Calcula el valor de Out en [IdBloque][IdThread], el cual para una matriz representada como array es [IdBloque*TamañoBloque+IdThread], y lo obtiene sumando el valor de a y b en ID
__global__ void AddMatrix(int* out, int* a, int* b) {
	int nThread = threadIdx.x;
	int nBlock = blockIdx.x;
	int blockDimension = blockDim.x;
	int id = nBlock * blockDimension + nThread;
	out[id] = a[id] + b[id];
}

//Mul Matrix toma dos matrices representadas como arrays y te devuelve otro array que representa el resultado de la multiplicación.
//En este caso, dado que toda multiplicacion de matrices funciona tal que A[X][Y]*B[Y][Z]= C[X][Z], la función se invoca con <<<NFilasdeA,NColumnasdeB>>>
//Con IdBloque e IdThread sabemos a que fila y columna pertenece el valor de Out, y como calcularlo. 
//Se recorre esa Fila de A y Columna de B, las cuales tienen el mismo tamaño (nColumns1), y se calcula la suma de las multiplicaciones de los valores. 
//Como A y B son vectores, para recorrer la fila de A se encuentra la "posición 0" de dicha fila y se le suma 1 por cada iteración para obtener el siguiente valor. 
//Para recorrer la columna de B, se encuentra la "posición 0" de dicha columna, y se le suma el tamaño de la fila de B para obtener el siguiente valor. 

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


//MoveMatrix es un Kernel de apoyo que uso para copiar una matriz a otra. No revisa tamaños, dado que trabaja con arrays. Debido a como está hecha, funciona con cualquier <<<X,Y>>> 
//Siempre que X*Y sea igual al número de elementos de la matriz 
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
	
	//Asegurarse de que el tamaño es el mismo
	/*Intento de calcular el tamaño dentro de la funcion: IMPOSIBLE
	size_t size1 = sizeof(vector1) / sizeof(vector1[0]);
	size_t size2 = sizeof(vector2) / sizeof(vector2[0]);
	printf("Size of vector1: %zd\n", size1);
	printf("Size of vector2: %zd\n", size2);
	*/
	if (size1 == size2) {
		//Declarar variables
		int* vOut = (int*)malloc(sizeof(int) * size1);
		int* d_v1, * d_v2, * d_vOut;

		//Hacer espacio en la GPU
		cudaMalloc((void**)&d_v1, sizeof(int) * size1);
		cudaMalloc((void**)&d_v2, sizeof(int) * size1);
		cudaMalloc((void**)&d_vOut, sizeof(int) * size1);

		//Pasar información al GPU
		cudaMemcpy(d_v1, vector1, sizeof(int) * size1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, vector2, sizeof(int) * size1, cudaMemcpyHostToDevice);

		//Invocar funcion de suma
		AddVector << <1, size1>> > (d_vOut, d_v1, d_v2);

		//Pasar información resultante de vuelta
		cudaMemcpy(vOut, d_vOut, sizeof(int) * size1, cudaMemcpyDeviceToHost);
		
		//Liberar el espacio usado en la GPU
		cudaFree(d_v1);
		cudaFree(d_v2);
		cudaFree(d_vOut);

		if (wantPrint) {
			//Print resultado
			for (int i = 0; i < size1; i++) {
				printf("%d + %d = %d\n", vector1[i], vector2[i], vOut[i]);
			}
		}
		

		
	}
	else {
		printf("Mismatch in vector sizes");
	}


}

//Como no hay manera de averiguar el numero de filas y columnas, si los tamaños cuadran trabaja el resultado en función de nrowsxncolumns
void SumaMatricial(int matriz1[], int matriz2[], int size1, int size2, int nRows, int nColumns, bool wantPrint) {

	//Asegurarse de que el tamaño es el mismo es imposible dentro de la función
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

		//Pasar información al GPU
		cudaMemcpy(d_v1, matriz1, sizeof(int) * expectedSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, matriz2, sizeof(int) * expectedSize, cudaMemcpyHostToDevice);

		//Invocar funcion de suma
		AddMatrix << <nRows, nColumns >> > (d_vOut, d_v1, d_v2);

		//Pasar información resultante de vuelta
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
	//Comprobar que los tamaños cuadran
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

			//Pasar información al GPU
			cudaMemcpy(d_v1, matriz1, sizeof(int) * size1, cudaMemcpyHostToDevice);
			cudaMemcpy(d_v2, matriz2, sizeof(int) * size2, cudaMemcpyHostToDevice);
			//Sleep(2000);
			//Invocar funcion de multiplicacion
			MulMatrix << <nRows1, nColumns2 >> > (d_vOut, d_v1, d_v2, nColumns1);

			//Pasar información resultante de vuelta
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
		printf("Los tamaños de las matrices no cuadran con las filas y columnas especificadas");
	}
}

void PotenciaMatricial(int matriz[], int size, int nRows, int nColumns, int nPotencia, bool wantPrint) {
	//Comprobar tamaño
	if (size == nRows * nColumns && nRows==nColumns) {
		//Declarar variables
		int* vOut = (int*)malloc(sizeof(int) * size);
		int* d_v1, * d_v2, * d_vOut;

		//Hacer espacio en el GPU
		cudaMalloc((void**)&d_v1, sizeof(int) * size);
		cudaMalloc((void**)&d_v2, sizeof(int) * size);
		cudaMalloc((void**)&d_vOut, sizeof(int) * size);

		//Pasar información al GPU
		cudaMemcpy(d_v1, matriz, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, matriz, sizeof(int) * size, cudaMemcpyHostToDevice);


		//For
		for (int i = 1; i < nPotencia; i++) {
			MulMatrix << <nRows, nColumns >> > (d_vOut, d_v1, d_v2, nColumns);
			MoveMatrix << <nRows, nColumns >> > (d_v1, d_vOut);
		}

		//Pasar información resultante de vuelta
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
		printf("Tamaño de Matriz incorrecto");
	}
}

int main() {

	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
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

	
	// Start record
	cudaEventRecord(start, 0);
	// Do something on GPU
	PotenciaMatricial(vector1, NELEMS(vector1), 4, 4, 3, true);
	// Stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Time Elapsed: %f", elapsedTime);

	return 0;
}