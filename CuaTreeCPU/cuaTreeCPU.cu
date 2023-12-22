#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <Windows.h>
#define N 5 //Número de cuerpos en el universo
#define TIMELAPSE 86400 //Número de segundos que pasan entre instantes
#define G 6.67428/100000000000

#define MAXDIM 100000 // m
#define MAXSPEED 60000 // m/s
#define MAXMASS 61000000
#define MINMASS 1000000
#define randnum(min, max) \
        ((rand() % (int)(((max) + 1) - (min))) + (min))



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

struct cuaTree {
	bool tieneRamas = false;
	int nivelTree;
	int cuadrante;
	int path;
	int* cuerpos;
	int nCuerpos;
	float masaTotal;
	float centroMasasX;
	float centroMasasY;
	float maxX;
	float maxY;
	float minX;
	float minY;
	struct cuaTree *rama0 = NULL;	//	0	1
	struct cuaTree *rama1 = NULL;	//	2	3
	struct cuaTree *rama2 = NULL;
	struct cuaTree *rama3 = NULL;
};



float randomPos() {
	float pos1 = 0 - (float) MAXDIM;
	float generablePos = (float) MAXDIM * 2 / 20000;
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
	float randomNumber = (float)randomNumber1000();
	float mass = randomNumber * generableMass + MINMASS;
	//printf("GenerableMass: %f\n", generableMass);
	//printf("RandomNumber: %f\n", randomNumber);
	//printf("%f\n", mass);

	return mass;
}

cuerpo inicializar( float posicion[2], float velocidad[2], float masa) {
	struct cuerpo* devuelto = (cuerpo*)malloc(sizeof(struct cuerpo));
	cuerpo a = devuelto[0];
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

universo crearUniversoAleatorio(universo* uni) {

	float vel[2];
	float pos[2];
	float masa;
	for (int i = 0; i < N; i++) {
		masa = randomMass();
		vel[0] = randomSpeed();
		vel[1] = randomSpeed();
		pos[0] = randomPos();
		pos[1] = randomPos();
		uni[0].cuerpos[i] = inicializar(pos, vel, masa);

	}
	return uni[0];
}

float centroMasasX(int arrayCuerpos[], int nCuerpos, universo* uni) {
	int punteroCuerpos;
	float devuelto = 0;
	float sumatorioMasas = 0;
	cuerpo a;
	for (int i = 0; i < nCuerpos; i++) {
		punteroCuerpos = arrayCuerpos[i];
		a = uni[0].cuerpos[punteroCuerpos];
		sumatorioMasas += a.masa;
		devuelto += a.pos[0]*a.masa;
	}
	devuelto = devuelto / sumatorioMasas;

	return devuelto;
}

float centroMasasY(int arrayCuerpos[], int nCuerpos, universo* uni) {
	int punteroCuerpos;
	float devuelto = 0;
	float sumatorioMasas = 0;
	cuerpo a;
	for (int i = 0; i < nCuerpos; i++) {
		punteroCuerpos = arrayCuerpos[i];
		a = uni[0].cuerpos[punteroCuerpos];
		sumatorioMasas += a.masa;
		devuelto += a.pos[1] * a.masa;
	}
	devuelto = devuelto / sumatorioMasas;

	return devuelto;
}

void printTree(cuaTree raiz) {

	printf("-----------IMPRESION DE TREE------------\n\n");
	printf("Nivel del Tree: %d\n", raiz.nivelTree);
	printf("Cuadrante del Tree: %d\n", raiz.cuadrante);
	printf("Path del Tree: %d\n", raiz.path);
	printf("\nNumero de cuerpos del Tree: %d\n", raiz.nCuerpos);
	printf("Id's de Cuerpos del Tree: ");
	for (int i = 0; i < raiz.nCuerpos - 1; i++) {
		printf("%d, ", raiz.cuerpos[i]);
	}
	printf("%d.\n", raiz.cuerpos[raiz.nCuerpos - 1]);
	printf("\nMasa total del Tree: %f\n", raiz.masaTotal);
	printf("Centro de Masas X: %f\n", raiz.centroMasasX);
	printf("Centro de Masas Y: %f\n", raiz.centroMasasY);
	printf("\nDimensiones del Tree:\n");
	printf("MaxX: %f\n", raiz.maxX);
	printf("MinX: %f\n", raiz.minX);
	printf("MaxY: %f\n", raiz.maxY);
	printf("MinY: %f\n", raiz.minY);

	if (!raiz.tieneRamas) {
		printf("Este arbol no ha sido ramificado\n");
	}
	else {
		bool noHayRama0 = raiz.rama0 == NULL;
		bool noHayRama1 = raiz.rama1 == NULL;
		bool noHayRama2 = raiz.rama2 == NULL;
		bool noHayRama3 = raiz.rama3 == NULL;
		printf("Este arbol ha sido ramificado\n");
		if (noHayRama0) {
			printf("El cuadrante 0 no tiene cuerpos\n");
		}
		else {
			printf("El cuadrante 0 si tiene cuerpos\n");
			printTree(raiz.rama0[0]);
		}
		if (noHayRama1) {
			printf("El cuadrante 1 no tiene cuerpos\n");
		}
		else {
			printf("El cuadrante 1 si tiene cuerpos\n");
			printTree(raiz.rama1[0]);
		}
		if (noHayRama2) {
			printf("El cuadrante 2 no tiene cuerpos\n");
		}
		else {
			printf("El cuadrante 2 si tiene cuerpos\n");
			printTree(raiz.rama2[0]);
		}
		if (noHayRama3) {
			printf("El cuadrante 3 no tiene cuerpos\n");
		}
		else {
			printf("El cuadrante 3 si tiene cuerpos\n");
			printTree(raiz.rama3[0]);
		}

	}


}

cuaTree* primerTree(universo* uni) {
	struct cuaTree* devuelto = (cuaTree*)malloc(sizeof(struct cuaTree));
	devuelto = new cuaTree;
	int* cuerpos = (int*)malloc(N * sizeof(int));
	float masaTotal = 0.0;
	for (int i = 0; i < N; i++) {
		cuerpos[i] = i;
		masaTotal += uni[0].cuerpos[i].masa;
		//printf("Masas del cuerpo: %f\n", uni[0].cuerpos[i].masa);
	}
	devuelto->path = 0;
	devuelto->cuadrante = 0;
	devuelto->nivelTree = 0;
	devuelto->nCuerpos = N;
	devuelto->cuerpos = cuerpos;
	devuelto->masaTotal = masaTotal;
	devuelto->centroMasasX = centroMasasX(cuerpos, N, uni);
	devuelto->centroMasasY = centroMasasY(cuerpos, N, uni);
	devuelto->maxX = (float)MAXDIM;
	devuelto->maxY = (float)MAXDIM;
	devuelto->minX = 0.0 - (float)MAXDIM;
	devuelto->minY = 0.0 - (float)MAXDIM;

	printTree(devuelto[0]);
	return devuelto;
}

int aQueCuadrante(cuerpo a, cuaTree raiz) { 
	//	0	1
	//	2	3
	int cuadrante = 0;

	float minX = raiz.minX;
	float maxX = raiz.maxX;
	float minY = raiz.minY;
	float maxY = raiz.maxY;

	float posX = a.pos[0];
	float posY = a.pos[1];

	float midX = (minX + maxX) / 2; 
	float midY = (minY + maxY) / 2;

	if (posX > midX) { //(midX,maxX] 
		cuadrante += 1;
	}
	if (posY <= midY) {//[minY, midY]
		cuadrante += 2;
	}
	//Con min = -10 y max = 10
	//0-> ( posX C [minX,midX], posY C (midY, maxY] ) -> ( posX C [-10,0], posY C  (0, 10] )
	//1-> ( posX C (midX,maxX], posY C (midY, maxY] ) -> ( posX C  (0,10], posY C  (0, 10] )
	//2-> ( posX C [minX,midX], posY C [minY, midY] ) -> ( posX C [-10,0], posY C [-10, 0] )
	//3-> ( posX C (midX,maxX], posY C [minY, midY] ) -> ( posX C  (0,10], posY C [-10, 0] )
	
	//Cuerpos en el Eje Y pertenecen a cuadrantes izquierdos
	//Cuerpos en el Eje X pertenecen a cuadrantes inferiores
	//Punto medio pertenece a cuadrante 2.
	return cuadrante;
}

void ramificaTree(cuaTree* raiz, universo* uni) {
	//VariablesDeApoyo
	raiz[0].tieneRamas = true;
	float minX = raiz[0].minX;
	float maxX = raiz[0].maxX;
	float minY = raiz[0].minY;
	float maxY = raiz[0].maxY;
	float midX = (minX + maxX) / 2;
	float midY = (minY + maxY) / 2;

	int nCuerposTotales = raiz[0].nCuerpos;

	int nCuerposRama0 = 0;
	int nCuerposRama1 = 0;
	int nCuerposRama2 = 0;
	int nCuerposRama3 = 0;

	int* cuerpos0 = (int*)malloc(nCuerposTotales * sizeof(int));
	int* cuerpos1 = (int*)malloc(nCuerposTotales * sizeof(int));
	int* cuerpos2 = (int*)malloc(nCuerposTotales * sizeof(int));
	int* cuerpos3 = (int*)malloc(nCuerposTotales * sizeof(int));

	float masaTotal0 = 0;
	float masaTotal1 = 0;
	float masaTotal2 = 0;
	float masaTotal3 = 0;

	int numCuerpo;
	int num;
	cuerpo cuerpoTree;
	//Recorrer cuerpos del tree
	for (int i = 0; i < nCuerposTotales; i++) {
		numCuerpo = raiz[0].cuerpos[i];
		cuerpoTree = uni[0].cuerpos[numCuerpo];
		num = aQueCuadrante(cuerpoTree, raiz[0]);

		switch (num) {
		case 0: cuerpos0[nCuerposRama0] = numCuerpo;
				nCuerposRama0++;
				masaTotal0 += cuerpoTree.masa;
				break;

		case 1: cuerpos1[nCuerposRama1] = numCuerpo;
				nCuerposRama1++;
				masaTotal1 += cuerpoTree.masa;
				break;

		case 2: cuerpos2[nCuerposRama2] = numCuerpo;
				nCuerposRama2++;
				masaTotal2 += cuerpoTree.masa;
				break;

		case 3: cuerpos3[nCuerposRama3] = numCuerpo;
				nCuerposRama3++;
				masaTotal3 += cuerpoTree.masa;
				break;
		}
		
	}
	
	//Rama 0 

	if (nCuerposRama0 > 0) {
		struct cuaTree* rama0 = (cuaTree*)malloc(sizeof(struct cuaTree));
		rama0 = new cuaTree;
		int* trueCuerpos0 = (int*)malloc(nCuerposRama0 * sizeof(int));
		for (int i = 0; i < nCuerposRama0; i++) {
			trueCuerpos0[i] = cuerpos0[i];
		}

		rama0->cuerpos = trueCuerpos0;
		rama0->path = raiz[0].path * 10 + 1;
		rama0->cuadrante = 1;
		rama0->nivelTree = raiz[0].nivelTree + 1;
		rama0->masaTotal = masaTotal0;
		rama0->maxX = midX;
		rama0->minX = minX;
		rama0->maxY = maxY;
		rama0->minY = midY;
		rama0->nCuerpos = nCuerposRama0;
		rama0->centroMasasX = centroMasasX(trueCuerpos0, nCuerposRama0, uni);
		rama0->centroMasasY = centroMasasY(trueCuerpos0, nCuerposRama0, uni);
		raiz[0].rama0 = rama0;
	}
	free(cuerpos0);

	//Rama 1 
	if (nCuerposRama1 > 0) {
		struct cuaTree* rama1 = (cuaTree*)malloc(sizeof(struct cuaTree));
		rama1 = new cuaTree;
		int* trueCuerpos1 = (int*)malloc(nCuerposRama1 * sizeof(int));
		for (int i = 0; i < nCuerposRama1; i++) {
			trueCuerpos1[i] = cuerpos1[i];
		}

		rama1->cuerpos = trueCuerpos1;
		rama1->path = raiz[0].path * 10 + 2;
		rama1->cuadrante = 2;
		rama1->nivelTree = raiz[0].nivelTree + 1;
		rama1->masaTotal = masaTotal1;
		rama1->maxX = maxX;
		rama1->minX = midX;
		rama1->maxY = maxY;
		rama1->minY = midY;
		rama1->nCuerpos = nCuerposRama1;
		rama1->centroMasasX = centroMasasX(trueCuerpos1, nCuerposRama1, uni);
		rama1->centroMasasY = centroMasasY(trueCuerpos1, nCuerposRama1, uni);
		raiz[0].rama1 = rama1;
	}
	free(cuerpos1);

	//Rama 2 
	if (nCuerposRama2 > 0) {//Si tiene cuerpos esa rama deja de ser un null
		struct cuaTree* rama2 = (cuaTree*)malloc(sizeof(struct cuaTree));
		rama2 = new cuaTree;
		int* trueCuerpos2 = (int*)malloc(nCuerposRama2 * sizeof(int));
		for (int i = 0; i < nCuerposRama2; i++) {
			trueCuerpos2[i] = cuerpos2[i];
		}

		rama2->cuerpos = trueCuerpos2;
		rama2->path = raiz[0].path * 10 + 3;
		rama2->cuadrante = 3;
		rama2->nivelTree = raiz[0].nivelTree + 1;
		rama2->masaTotal = masaTotal2;
		rama2->maxX = midX;
		rama2->minX = minX;
		rama2->maxY = midY;
		rama2->minY = maxY;
		rama2->nCuerpos = nCuerposRama2;
		rama2->centroMasasX = centroMasasX(trueCuerpos2, nCuerposRama2, uni);
		rama2->centroMasasY = centroMasasY(trueCuerpos2, nCuerposRama2, uni);
		raiz[0].rama2 = rama2;
	}
	free(cuerpos2);

	//Rama 3 
	if (nCuerposRama3 > 0) {
		struct cuaTree* rama3 = (cuaTree*)malloc(sizeof(struct cuaTree));
		rama3 = new cuaTree;
		int* trueCuerpos3 = (int*)malloc(nCuerposRama3 * sizeof(int));
		for (int i = 0; i < nCuerposRama3; i++) {
			trueCuerpos3[i] = cuerpos3[i];
		}
		rama3->cuerpos = trueCuerpos3;
		rama3->path = raiz[0].path * 10 + 4;
		rama3->cuadrante = 4;
		rama3->nivelTree = raiz[0].nivelTree + 1;
		rama3->masaTotal = masaTotal3;
		rama3->maxX = maxX;
		rama3->minX = midX;
		rama3->maxY = midY;
		rama3->minY = minY;
		rama3->nCuerpos = nCuerposRama3;
		rama3->centroMasasX = centroMasasX(trueCuerpos3, nCuerposRama3, uni);
		rama3->centroMasasY = centroMasasY(trueCuerpos3, nCuerposRama3, uni);
		raiz[0].rama3 = rama3;
	}
	free(cuerpos3);

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
	cuerpo cuerpoActual;
	for (int i = 0; i < N; i++) {
		cuerpoActual = uni[0].cuerpos[i];
		fuerzaX = cuerpoActual.fuerzas[0];
		fuerzaY = cuerpoActual.fuerzas[1];
		masa = cuerpoActual.masa;

		cuerpoActual.acel[0] = fuerzaX / masa;
		cuerpoActual.acel[1] = fuerzaY / masa;
		
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
	}
	if (print) {
		printCuerpos(uni, nIteration, true, true);
		writeData(uni, nIteration, nIteracionesTotales+1);
	}
}

int main() {
	struct universo* uni = (universo*)malloc(sizeof(universo));
	struct cuaTree* primerArbol;
	float posicion[] = { 0,0 };
	float posicion2[] = { 12,8 };
	float posicion3[] = { 12,-14 };
	float posicion4[] = { -14,-14 };
	float posicion5[] = { -16,12 };
	float velocidad[] = { 0,0 };
	float masa = 10000000000.0;//Problemas surgen entre 10^10 y 5*10^10

;
	uni[0].cuerpos[0] = inicializar(posicion, velocidad, masa);
	uni[0].cuerpos[1] = inicializar(posicion2, velocidad, masa);
	uni[0].cuerpos[2] = inicializar(posicion3, velocidad, masa);
	uni[0].cuerpos[3] = inicializar(posicion4, velocidad, masa);
	uni[0].cuerpos[4] = inicializar(posicion5, velocidad, masa);
	//uni[0] = crearUniversoAleatorio(uni);
	primerArbol = primerTree(uni);
	ramificaTree(primerArbol, uni);
	printTree(primerArbol[0]);
	return 0;
}
