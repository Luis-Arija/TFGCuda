#include <stdio.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <Windows.h>
#define N 5 //Número de cuerpos en el universo
#define TIMELAPSE 1 //Número de segundos que pasan entre instantes
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))


/*	Orden:
		Nueva Fuerza:
			->Obtener la distancia ortogonal entre A y B
			->Obtener la distancia entre A y B
			->Obtener fuerza diagonal 
			->Calcular fuerza ortogonal en base a la fuerza diagonal
			->Obtener Matriz de fuerzas
			->Sumar Matriz de fuerzas
		Nueva Aceleración DONE
		Nueva Posición DONE
		Nueva Velocidad DONE
*/



//tamaño cuerpo = 36
struct cuerpo {
	float pos[2];	//En metros
	float vel[2];	//En Metros/Segundo
	float masa;		//En KG
	float acel[2];	//En Metros/Segundo^2 = F/M
	float fue[2];	//En Newtons (N) = G* (m1*m2) / d^2
};

//Tamaño universo = N*36 = N*Tamaño_Cuerpo
struct universo {
	struct cuerpo cuerpos[N];
};

//Mete los datos en cuerpo
cuerpo inicializar(cuerpo a, float posicion[2], float velocidad[2], float masa) {
	a.pos[0] = posicion[0];
	a.pos[1] = posicion[1];
	a.vel[0] = velocidad[0];
	a.vel[1] = velocidad[1];
	a.masa = masa;
	a.acel[0] = 0; 
	a.acel[1] = 0;
	a.fue[0] = 0;
	a.fue[1] = 0;
	return a;
}

int nCalculos(int ncuerpos) {
	int sumatorio = 0;
	for (int i = ncuerpos; i > 1; i--) {
		sumatorio += i - 1;
	}
	return sumatorio;
}

//Tantos bloques como objetos, un thread por dimension. El codigo no es óptimo en pos de ser representativo. 
__global__ void newAcel(universo* uni) {

	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;

	//Obtener Masa y Fuerza actual de esta dimension
	float masa_actual = uni[0].cuerpos[nBlock].masa;
	float fue_actual = uni[0].cuerpos[nBlock].fue[nThread];

	//Calcular la nueva Aceleración y meterla en el universo
	float acel_nueva = fue_actual/masa_actual;
	uni[0].cuerpos[nBlock].pos[nThread] = acel_nueva;
}

//Tantos bloques como objetos, un thread por dimension. El codigo no es óptimo en pos de ser representativo. 
__global__ void newPosition(universo* uni) {

	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;
	
	//Obtener Posicion y Velocidad actual de esta dimension
	float pos_actual = uni[0].cuerpos[nBlock].pos[nThread];
	float vel_actual = uni[0].cuerpos[nBlock].vel[nThread];

	//Calcular la nueva Posición y meterla en el universo
	float pos_nueva = pos_actual + vel_actual * TIMELAPSE;
	printf("Posicion nueva: %f\n", pos_nueva);
	uni[0].cuerpos[nBlock].pos[nThread] = pos_nueva;
}

//Tantos bloques como objetos, un thread por dimension. El codigo no es óptimo en pos de ser representativo. 

__global__ void newSpeed(universo* uni) {

	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;

	//Obtener Velocidad y Aceleración actual de esta dimension
	float vel_actual = uni[0].cuerpos[nBlock].vel[nThread];
	float acel_actual = uni[0].cuerpos[nBlock].acel[nThread];

	//Calcular la nueva Velocidad y meterla en el universo
	float vel_nueva = vel_actual + acel_actual * TIMELAPSE;
	uni[0].cuerpos[nBlock].vel[nThread] = vel_nueva;
}

__global__ void printUni(universo* uni) {
	printf("Posicion = %f, %f\n", uni[0].cuerpos[0].pos[0], uni[0].cuerpos[0].pos[1]);
	printf("Velocidad = %f, %f\n", uni[0].cuerpos[0].vel[0], uni[0].cuerpos[0].vel[1]);
}

void iterar_universo(universo uni, int tiempo, bool print) {
	universo* d_uni;
	cudaMalloc(&d_uni, sizeof(universo));
	
	cudaMemcpy(d_uni, &uni, sizeof(universo), cudaMemcpyHostToDevice);
	//Print situación inicial
	printUni << <1, 1 >> > (d_uni);
	for (int i = 0; i <= tiempo; i = i + TIMELAPSE) {
	
		//Obtener fuerzas


		newAcel << <N, 2 >> > (d_uni);
		newPosition << <N, 2 >> > (d_uni);
		newSpeed << <N, 2 >> > (d_uni);

		//Print situación T=i+Timelapse
		printUni << <1, 1 >> > (d_uni);
	}
	
	cudaFree(d_uni);
}

int main() {
	
	//printf("Tamaño cuerpo: %d\n", sizeof(universo));
	
	struct cuerpo mundo1;
	float posicion[] = { 0,0 };
	float velocidad[] = { 1,1 };
	float masa = 10;

	mundo1 = inicializar(mundo1, posicion, velocidad, masa);

	struct universo uni; 
	uni.cuerpos[0] = inicializar(mundo1, posicion, velocidad, masa);
	uni.cuerpos[1] = inicializar(mundo1, posicion, velocidad, masa);

	iterar_universo(uni, 2, true);

	return 0;
}
