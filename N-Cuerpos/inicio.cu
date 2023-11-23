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
	float fueTotal[2];	
	float fueVarias[N][2]; //En Newtons (N) = G* (m1*m2) / d^2
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
	return a;
}

int nCalculos(int ncuerpos) {
	int sumatorio = 0;
	for (int i = ncuerpos; i > 1; i--) {
		sumatorio += i - 1;
	}
	return sumatorio;
}

__global__ void force0(universo* uni) {
	for (int i = 0; i < N; i++) {
		uni[0].cuerpos[i].fueVarias[i][0] = 0;
		uni[0].cuerpos[i].fueVarias[i][1] = 0;
	}
}

__global__ void newVariousForce(universo* uni) {
	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;

	if (nBlock < nThread) {
		float posX1 = uni[0].cuerpos[nBlock].pos[0];
		float posY1 = uni[0].cuerpos[nBlock].pos[1];
		float posX2 = uni[0].cuerpos[nThread].pos[0];
		float posY2 = uni[0].cuerpos[nThread].pos[1];
		
		float M1 = uni[0].cuerpos[nBlock].masa;
		float M2 = uni[0].cuerpos[nBlock].masa;
		
		float difX = posX1 - posX2;
		float difY = posY1 - posY2;

		float disTotal = sqrt(difX * difX + difY * difY);
		
		float Div = 100000000000*disTotal;
		float F = 6.67428 * M1 * M2 / Div;
		
		float cos = difX / disTotal;
		float sen = difY / disTotal;

		float Fx = F * cos;
		float Fy = F * sen;
		
		uni[0].cuerpos[nBlock].fueVarias[nThread][0] = -Fx;
		uni[0].cuerpos[nBlock].fueVarias[nThread][1] = -Fy;

		uni[0].cuerpos[nThread].fueVarias[nBlock][0] = Fx;
		uni[0].cuerpos[nThread].fueVarias[nBlock][1] = Fy;

	}

}

//Tantos bloques como objetos, un thread por dimension. El codigo no es óptimo en pos de ser representativo. 
__global__ void newForce(universo* uni) {
	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;
	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += uni[0].cuerpos[nBlock].fueVarias[i][nThread];
	}
	uni[0].cuerpos[nBlock].fueTotal[nThread] = sum;

}

//Tantos bloques como objetos, un thread por dimension. El codigo no es óptimo en pos de ser representativo. 
__global__ void newAcel(universo* uni) {

	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;

	//Obtener Masa y Fuerza actual de esta dimension
	float masa_actual = uni[0].cuerpos[nBlock].masa;
	float fue_actual = uni[0].cuerpos[nBlock].fueTotal[nThread];
	//Calcular la nueva Aceleración y meterla en el universo
	float acel_nueva = fue_actual/masa_actual;
	uni[0].cuerpos[nBlock].acel[nThread] = acel_nueva;
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
	printf("Pos X		Pos Y		VelX		VelY\n");
	for (int i = 0; i < N; i++) {
		printf("%f, %f,", uni[0].cuerpos[i].pos[0], uni[0].cuerpos[i].pos[1]);
		printf("	%f, %f\n", uni[0].cuerpos[i].vel[0], uni[0].cuerpos[i].vel[1]);
	}
}

void iterar_universo(universo uni, int tiempo, bool print) {
	universo* d_uni;
	cudaMalloc(&d_uni, sizeof(universo));
	cudaMemcpy(d_uni, &uni, sizeof(universo), cudaMemcpyHostToDevice);
	
	force0 << <1, 1 >> > (d_uni);
	printf("Iteracion 0: \n");
	printUni << <1, 1 >> > (d_uni);
	for (int i = 0; i <= tiempo; i = i + TIMELAPSE) {
	
		printf("Iteracion %d: \n", (i+1));
		//Obtener fuerzas
		newVariousForce << <N, N >> > (d_uni);
		newForce << <N, 2 >> > (d_uni);
		newAcel << <N, 2 >> > (d_uni);
		newPosition << <N, 2 >> > (d_uni);
		newSpeed << <N, 2 >> > (d_uni);
		//Print situación T=i+Timelapse

		printUni << <1, 1 >> > (d_uni);

		cudaDeviceSynchronize();
	}
	cudaFree(d_uni);
}

int main() {
	
	//printf("Tamaño cuerpo: %d\n", sizeof(universo));
	
	struct cuerpo mundo1;
	float posicion[] = { 0,0 };
	float posicion2[] = { 10,10 };
	float posicion3[] = { 10,-10 };
	float posicion4[] = { -10,-10 };
	float posicion5[] = { -10,10 };
	float velocidad[] = { 0,0 };
	float masa = 1000000000000;

	struct universo uni;
	uni.cuerpos[0] = inicializar(mundo1, posicion, velocidad, masa);
	uni.cuerpos[1] = inicializar(mundo1, posicion2, velocidad, masa);
	uni.cuerpos[2] = inicializar(mundo1, posicion3, velocidad, masa);
	uni.cuerpos[3] = inicializar(mundo1, posicion4, velocidad, masa);
	uni.cuerpos[4] = inicializar(mundo1, posicion5, velocidad, masa);
	iterar_universo(uni, 20, true);

	return 0;
}
