#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <Windows.h>
#define N 10//Número de cuerpos en el universo. MAXIMO 14000
#define nNiveles 1 //Número de niveles. Máximo 9 por tamaño int
//tendrán una dim de MAXDIM/pow(2,nNiveles)

#define TIMELAPSE 86400 //Número de segundos que pasan entre instantes
#define G 6.67428/100000000000
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))
#define MAXDIM 15*pow(10, 8)
#define MAXSPEED 30000 // m/s
#define MAXMASS 6*pow(10,24)
#define MINMASS 1*pow(10,23)
#define CLEANTREEITERATION 5
#define randnum(min, max) \
        ((rand() % (int)(((max) + 1) - (min))) + (min))

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



//tamaño cuerpo =
struct cuerpo {
	float pos[2];	//En metros, modificado mediante sumas y restas
	float vel[2];	//En Metros/Segundo, modificado mediante sumas y restas
	float masa;		//En KG, estático
	float acel[2];	//En m/s^2, cada iteración es nuevo
	float fuerzas[2]; //En N, cada iteración es nuevo.
};

//Tamaño universo = N*36 = N*Tamaño_Cuerpo
struct universo {
	struct cuerpo cuerpos[N];
};

cuerpo inicializar(cuerpo a, float posicion[2], float velocidad[2], float masa) {
	a.pos[0] = posicion[0];
	a.pos[1] = posicion[1];
	a.vel[0] = velocidad[0];
	a.vel[1] = velocidad[1];
	a.acel[0] = 0;
	a.acel[1] = 0;
	a.fuerzas[0] = 0;
	a.fuerzas[1] = 0;
	a.masa = masa;
	return a;
}

int randomNumber1000(){
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
	float pos1 = 0-MAXDIM;
	float generablePos = (MAXDIM)*2 / 20000;
	float randomNumber = (float)randomNumber20000();
	float pos = pos1 + randomNumber * generablePos;

	return pos;
}

float randomSpeed() {
	float speed1 = 0 - MAXSPEED;
	float generableSpeed = (MAXSPEED) * 2 / 20000;
	float randomNumber = (float)randomNumber20000();
	float speed = speed1 + randomNumber * generableSpeed;

	//printf("%f\n", speed);

	return speed;
}

float randomMass() {
	float generableMass = (MAXMASS - MINMASS) / 1000;
	float randomNumber = (float) randomNumber1000();
	float mass = randomNumber * generableMass + MINMASS;
	//printf("GenerableMass: %f\n", generableMass);
	//printf("RandomNumber: %f\n", randomNumber);
	//printf("%f\n", mass);
	
	return mass;
}

//Le pasas un puntero malloc y lo rellena
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

	//return uni[0];
}

void forceIterate(universo* uni, int idCuerpo1, int idCuerpo2) {

	cuerpo cuerpo1 = uni[0].cuerpos[idCuerpo1];
	cuerpo cuerpo2 = uni[0].cuerpos[idCuerpo2];

	float posX1 = cuerpo1.pos[0];
	float posY1 = cuerpo1.pos[1];
	float posX2 = cuerpo2.pos[0];
	float posY2 = cuerpo2.pos[1];

	float M1 = cuerpo1.masa;
	float M2 = cuerpo2.masa;

	float difX = posX1 - posX2;
	float difY = posY1 - posY2;

	float disTotal = sqrt(difX * difX + difY * difY);

	float F = G * M1 * M2 / (disTotal * disTotal);

	float cos = difX / disTotal;
	float sen = difY / disTotal;

	float Fx = F * cos;
	float Fy = F * sen;

	cuerpo1.fuerzas[0] -= Fx;
	cuerpo1.fuerzas[1] -= Fy;

	cuerpo2.fuerzas[0] += Fx;
	cuerpo2.fuerzas[1] += Fy;

	uni[0].cuerpos[idCuerpo1] = cuerpo1;
	uni[0].cuerpos[idCuerpo2] = cuerpo2;
}

void newForces(universo* uni) {
	//Las fuerzas pasan a ser 0
	for (int i = 0; i < N; i++) {
		uni[0].cuerpos[i].fuerzas[0] = 0;
		uni[0].cuerpos[i].fuerzas[1] = 0;
	}
	for (int i = 0; i < N; i++) {
		for (int j = i + 1; j < N; j++) {
			forceIterate(uni, i, j);
		}
	}
}

void newAcel(universo* uni) {
	float fuerzaX;
	float fuerzaY;
	float masa;
	float acelX;
	float acelY;
	cuerpo cuerpoActual;
	for (int i = 0; i < N; i++) {
		cuerpoActual = uni[0].cuerpos[i];
		fuerzaX = cuerpoActual.fuerzas[0];
		fuerzaY = cuerpoActual.fuerzas[1];
		masa = cuerpoActual.masa;
		acelX = fuerzaX / masa;
		acelY = fuerzaY / masa;
		cuerpoActual.acel[0] = acelX;
		cuerpoActual.acel[1] = acelY;
		
		uni[0].cuerpos[i] = cuerpoActual;
	}
}

void newPosition(universo * uni) {
	float velX;
	float velY;
	cuerpo cuerpoActual;

	for (int i = 0; i < N; i++) {
		
		cuerpoActual = uni[0].cuerpos[i];
		velX = cuerpoActual.vel[0];
		velY = cuerpoActual.vel[1];

		cuerpoActual.pos[0] += velX*TIMELAPSE;
		cuerpoActual.pos[1] += velY*TIMELAPSE;

		uni[0].cuerpos[i] = cuerpoActual;
	}
}

void newSpeed (universo* uni) {
	float acelX;
	float acelY;
	cuerpo cuerpoActual;

	for (int i = 0; i < N; i++) {
		cuerpoActual = uni[0].cuerpos[i];
		acelX = cuerpoActual.acel[0];
		acelY = cuerpoActual.acel[1];

		cuerpoActual.vel[0] += acelX * TIMELAPSE;
		cuerpoActual.vel[1] += acelY * TIMELAPSE;

		uni[0].cuerpos[i] = cuerpoActual;
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
	const char* nombreArchivo = "archivo.txt";
	if (iteracion == 0) {
		// Abrir el archivo en modo escritura ("w")
		archivo = fopen(nombreArchivo, "w");
		fprintf(archivo, "%d;%d", nIteracionesTotales, N);
	} else {
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

void iterateUniverse(universo* uni, int nSegundos, bool print) {
	int timeLeft = nSegundos;
	int nIteration = 0;
	int nIteracionesTotales = nSegundos / TIMELAPSE;
	while (timeLeft >= TIMELAPSE) {
		if (print) {
			printCuerpos(uni, nIteration, true, true);
			writeData(uni, nIteration, nIteracionesTotales+1);
		}
		newForces(uni);
		newAcel(uni);
		newPosition(uni);
		newSpeed(uni);
		timeLeft -= TIMELAPSE; 
		nIteration++;
		printf("llego a antes de newAccel\n");
		printf("Iteracion n: %d\n", nIteration);
	}
	if (print) {
		printCuerpos(uni, nIteration, true, true);
		writeData(uni, nIteration, nIteracionesTotales+1);
	}
}

int main() {

	//printf("Tamaño cuerpo: %d\n", sizeof(universo));

	struct universo* uni = (universo*)malloc(sizeof(universo));
	uni = new universo;
	crearUniversoAleatorio(uni);
	printCuerpos(uni, 0, true, true);
	iterateUniverse(uni, 864000, false);
	//printCuerpos(uni, 11, true, true);

	return 0;
}
