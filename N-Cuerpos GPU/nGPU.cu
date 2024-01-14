#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <Windows.h>


#define N 1024					//Número de Cuerpos en el universo
#define TIMELAPSE 3600			//Número de segundos que pasan entre instantes
#define G 6.67428/pow(10, 11)	//Constante G
#define MAXDIM 15*pow(10, 8)	//Rango de posición en X e Y
#define MAXSPEED 3*pow(10,3)	//Rango de velocidad en X e Y
#define MAXMASS 6*pow(10,24)	//Masa máxima de un Cuerpo
#define MINMASS 1*pow(10,23)	//Masa minima de un Cuerpo


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



//Tamaño Cuerpo = 36 + 8*N

struct cuerpo {
	float pos[2];			//Array de posición en Metros
	float vel[2];			//Array de velocidad n Metros/Segundo
	float masa;				//Masa del cuerpo en KG
	float acel[2];			//Array de aceleración en Metros/Segundo^2
	float fuerzas[2];		//Array de fuerzas en Newtons (N)
	float nFuerzas[N][2];	//Matriz de fuerzas en Newtons (N)
};

//Tamaño Universo CPU = 36*N
//Tamaño Universo GPU = 8*N^2 + 36*N
struct universo {
	struct cuerpo cuerpos[N];//Array de N Cuerpos
};

int randomNumber1000() {
	int randomNumber;
	bool checker = true;
	while (checker) {
		randomNumber = rand();
		if (randomNumber <= 1000) {
			checker = false;
		}
	}
	return randomNumber;
}
int randomNumber20000() {
	int randomNumber;
	bool checker = true;
	while (checker) {
		randomNumber = rand();
		if (randomNumber <= 20000) {
			checker = false;
		}
	}
	return randomNumber;
}
float randomPos() {
	float pos1 = 0 - MAXDIM;
	float generablePos = (MAXDIM) * 2 / 20000;
	float randomNumber = (float)randomNumber20000();
	float pos = pos1 + randomNumber * generablePos;
	return pos;
}
float randomSpeed() {
	float speed1 = 0 - MAXSPEED;
	float generableSpeed = (MAXSPEED) * 2 / 1000;
	float randomNumber = (float)randomNumber1000();
	float speed = speed1 + randomNumber * generableSpeed;
	return speed;
}
float randomMass() {
	float generableMass = (MAXMASS - MINMASS) / 20000;
	float randomNumber = (float)randomNumber20000();
	float mass = randomNumber * generableMass + MINMASS;
	return mass;

}
void crearUniversoAleatorio(universo* uni) {
	struct cuerpo a;
	for (int i = 0; i < N; i++) {
		a = uni->cuerpos[i];
		a.masa = randomMass();
		a.vel[0] = randomSpeed();
		a.vel[1] = randomSpeed();
		a.pos[0] = randomPos();
		a.pos[1] = randomPos();
		a.acel[0] = 0;
		a.acel[1] = 0;
		a.fuerzas[0] = 0;
		a.fuerzas[1] = 0;
		uni->cuerpos[i] = a;
	}
}
void printCuerpos(universo* uni, int iteracion, bool position, bool speed) {
	cuerpo cuerpoActual;
	printf("-------- ITERACION %d --------\n\n", iteracion);
	for (int i = 0; i < N; i++) {
		cuerpoActual = uni[0].cuerpos[i];
		printf("Cuerpo %d:\n\n", i);
		if (position) {
			printf("--Posicion:\n	X:%f\n	Y:%f\n\n", cuerpoActual.pos[0], cuerpoActual.pos[1]);
		}
		if (speed) {
			printf("--Speed:\n	X:%f\n	Y:%f\n\n", cuerpoActual.vel[0], cuerpoActual.vel[1]);
		}

	}
}
void writeData(universo* uni, int iteracion, int nIteracionesTotales) {
	cuerpo cuerpoActual;
	float posX;
	float posY;
	FILE* archivo;
	// Nombre del archivo
	const char* nombreArchivo = "Resultados nCuerposGPU.txt";
	if (iteracion == 0) {
		// Abrir el archivo en modo escritura ("w")
		archivo = fopen(nombreArchivo, "w");
		fprintf(archivo, "%d;%d", nIteracionesTotales, N);
	}
	else {
		// Abrir el archivo en modo adición ("a")
		archivo = fopen(nombreArchivo, "a");
	}

	for (int i = 0; i < N; i++) {
		//Obtener datos
		cuerpoActual = uni[0].cuerpos[i];
		posX = cuerpoActual.pos[0];
		posY = cuerpoActual.pos[1];

		fprintf(archivo, "\n%d;%d;%f;%f", iteracion, i, posX, posY);
		//fprintf(archivo, "\n%f;%f", posX, posY);
		//Imprimir en formato X;Y
	}

	fclose(archivo);


}




__global__ void force0(universo* uni) {
	for (int i = 0; i < N; i++) {
		uni[0].cuerpos[i].nFuerzas[i][0] = 0;
		uni[0].cuerpos[i].nFuerzas[i][1] = 0;
	}
}

__global__ void newNForcesGPU(universo* uni) {
	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;

	if (nBlock < nThread) {
		
		float posX1 = uni[0].cuerpos[nBlock].pos[0];
		float posY1 = uni[0].cuerpos[nBlock].pos[1];
		float posX2 = uni[0].cuerpos[nThread].pos[0];
		float posY2 = uni[0].cuerpos[nThread].pos[1];
		
		float M1 = uni[0].cuerpos[nBlock].masa;
		float M2 = uni[0].cuerpos[nThread].masa;
		
		float difX = posX1 - posX2;
		float difY = posY1 - posY2;

		float disTotal = sqrt(difX * difX + difY * difY);

		float Div = 100000000000 * disTotal * disTotal;
		float F = 6.67428 * M1 * M2 / Div;
		
		float cos = difX / disTotal;
		float sen = difY / disTotal;

		float Fx = F * cos;
		float Fy = F * sen;
		
		uni[0].cuerpos[nBlock].nFuerzas[nThread][0] = -Fx;
		uni[0].cuerpos[nBlock].nFuerzas[nThread][1] = -Fy;

		uni[0].cuerpos[nThread].nFuerzas[nBlock][0] = Fx;
		uni[0].cuerpos[nThread].nFuerzas[nBlock][1] = Fy;

	}

} 
__global__ void newForceGPU(universo* uni) {
	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;
	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += uni[0].cuerpos[nBlock].nFuerzas[i][nThread];
	}
	uni[0].cuerpos[nBlock].fuerzas[nThread] = sum;

}
__global__ void newAcelGPU(universo* uni) {

	//Obtener bloque y thread
	int nBlock = blockIdx.x;
	int nThread = threadIdx.x;

	//Obtener Masa y Fuerza actual de esta dimension
	float masa_actual = uni[0].cuerpos[nBlock].masa;
	float fue_actual = uni[0].cuerpos[nBlock].fuerzas[nThread];
	//Calcular la nueva Aceleración y meterla en el universo
	float acel_nueva = fue_actual/masa_actual;
	uni[0].cuerpos[nBlock].acel[nThread] = acel_nueva;
} 
__global__ void newPositionGPU(universo* uni) {

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
__global__ void newSpeedGPU(universo* uni) {

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


void iterteUniverseGPU(universo* uni, int nSegundos, bool print) {
	int timeLeft = nSegundos;
	int nIteration = 0;
	int nIteracionesTotales = nSegundos / TIMELAPSE;
	universo* d_uni;
	cudaMalloc(&d_uni, sizeof(universo));
	cudaMemcpy(d_uni, uni, sizeof(universo), cudaMemcpyHostToDevice);

	while (timeLeft >= TIMELAPSE) {
		//PRINT
		if (print) {
			cudaMemcpy(uni, d_uni, sizeof(universo), cudaMemcpyDeviceToHost);
			writeData(uni, nIteration, nIteracionesTotales + 1);
		}

		newNForcesGPU << <N, N >> > (d_uni);
		newForceGPU << <N, 2 >> > (d_uni);
		newAcelGPU << <N, 2 >> > (d_uni);
		newPositionGPU << <N, 2 >> > (d_uni);
		newSpeedGPU << <N, 2 >> > (d_uni);
		timeLeft -= TIMELAPSE;
		nIteration++;
		cudaDeviceSynchronize();
	}
	if (print) {
		cudaMemcpy(uni, d_uni, sizeof(universo), cudaMemcpyDeviceToHost);
		writeData(uni, nIteration, nIteracionesTotales + 1);
	}
	cudaFree(d_uni);
}

int main() {

	clock_t tiempo_inicio, tiempo_final;
	double segundos;
	int tiempoIteracion = 36000;

	struct universo* uni = (universo*)malloc(sizeof(universo));
	uni = new universo;
	crearUniversoAleatorio(uni); //Rellena uni
	
	printf("Comienzo de la iteracion del universo\n");
	printf("	Numero de cuerpos:		%d\n", N);
	printf("	Segundos por iteracion:		%d\n", TIMELAPSE);
	printf("	Tiempo a iterar:		%d\n", tiempoIteracion);
	printf("	Numero de iteraciones:		%d\n", tiempoIteracion / TIMELAPSE);

	tiempo_inicio = clock();
	iterteUniverseGPU(uni, tiempoIteracion, true);
	tiempo_final = clock();

	segundos = (double)(tiempo_final - tiempo_inicio) / CLOCKS_PER_SEC; /*según que estes midiendo el tiempo en segundos es demasiado grande*/

	printf("\nTIEMPO TARDADO: %f\n", segundos);
	return 0;
}
