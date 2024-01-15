#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <Windows.h>

#define N 100					//Número de Cuerpos en el universo
#define TIMELAPSE 3600			//Número de segundos que pasan entre instantes
#define G 6.67428/pow(10, 11)	//Constante G
#define MAXDIM 15*pow(10, 8)	//Rango de posición en X e Y
#define MAXSPEED 3*pow(10,3)	//Rango de velocidad en X e Y
#define MAXMASS 6*pow(10,24)	//Masa máxima de un Cuerpo
#define MINMASS 1*pow(10,23)	//Masa minima de un Cuerpo
#define nNiveles 8				//Número de niveles del arbol. 
#define SIZE pow(2,nNiveles)	//Número de filas y columnas de la matriz de ramas final
#define CLEANTREEITERATION 1	//Número de iteraciones antes de actualizar el arbol


//STRUCTS
struct cuerpo {
	float pos[2];	//En metros, modificado mediante sumas y restas
	float vel[2];	//En Metros/Segundo, modificado mediante sumas y restas
	float masa;		//En KG, estático
	float acel[2];	//En m/s^2, cada iteración es nuevo
	float fuerzas[2]; //En N, cada iteración es nuevo.
};

struct cuaTree {
	bool tieneRamas = false;
	int nivelTree=0; 
	int cuadrante=0; 
	int path=0; 
	int idFila=-1;
	int idColumna=-1;
	int nCuerpos = 0;
	int* cuerpos;
	float masaTotal = 0.0;
	float centroMasasX = 0.0;
	float centroMasasY = 0.0;
	float maxX = 0.0;
	float maxY = 0.0;
	float minX = 0.0;
	float minY = 0.0;
	struct cuaTree *rama1 = NULL;	
	struct cuaTree *rama2 = NULL;
	struct cuaTree *rama3 = NULL;
	struct cuaTree *rama4 = NULL;
	struct cuaTree* padre = NULL;
};

struct universo {
	struct cuerpo cuerpos[N];		//Array de N cuerpos
	struct cuaTree*** punTreeMatriz;//Matriz de punteros a las Ramas finales
	};


//Si no uso punteros a los niveles más bajos, tendré que nadar a traves de las ramas en ambas direcciones. 
//Ejemplo, un tree a la izquierda de un tree de cuadrante 1 es su primo. Para alcanzarle, tengo que ir al abuelo del original, escoger el hijo adecuado, 
// y el hijo de ese hijo. Todo para un tree que está más cerca que su hermano del cuadrante 4
//Por ello, punteros. 
//Problema: 
//Solución: 

//GENERADORES DE NUMEROS ALEATORIOS

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

	//punTree
	int nFilas = (int) pow(2, nNiveles) + 1e-9;
	int nColumnas = (int) pow(2, nNiveles) + 1e-9;

	uni->punTreeMatriz = (cuaTree***)malloc(sizeof(cuaTree**) * nFilas);
	for (int i = 0; i < nFilas; i++) {
		uni->punTreeMatriz[i]=(cuaTree**)malloc(sizeof(cuaTree*)* nColumnas);
	}

	//return uni[0];
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
	const char* nombreArchivo = "Resultados TreeCPU.txt";
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
void printTree(cuaTree* raiz) {

	if (raiz == NULL) {
		//printf("Este espacio no contiene nada\n");
	}
	else {
		int path = raiz->path;
		int nFila = raiz->idFila;
		int nColumna = raiz->idColumna;
		printf("-----------IMPRESION DE TREE------------\n\n");
		printf("Nivel del Tree: %d\n", raiz->nivelTree);
		printf("Cuadrante del Tree: %d\n", raiz->cuadrante);
		printf("Path del Tree: %d\n", path);
		if (raiz->nivelTree == nNiveles) {
			printf("Este tree es un tree final\n");
			printf("Posicion verdadera del Tree: (%d,%d)\n", (nFila + 1), (nColumna + 1));
		}

		printf("\nNumero de cuerpos del Tree: %d\n", raiz->nCuerpos);
		printf("Id's de Cuerpos del Tree: ");
		for (int i = 0; i < raiz->nCuerpos - 1; i++) {
			printf("%d, ", raiz->cuerpos[i]);
		}
		printf("%d.\n", raiz->cuerpos[raiz->nCuerpos - 1]);
		printf("\nMasa total del Tree: %f\n", raiz->masaTotal);
		printf("Centro de Masas X: %f\n", raiz->centroMasasX);
		printf("Centro de Masas Y: %f\n", raiz->centroMasasY);
		printf("\nDimensiones del Tree:\n");
		printf("MaxX: %f\n", raiz->maxX);
		printf("MinX: %f\n", raiz->minX);
		printf("MaxY: %f\n", raiz->maxY);
		printf("MinY: %f\n", raiz->minY);

		if (!raiz->tieneRamas) {
			printf("Este arbol no ha sido ramificado\n");
		}
		else {
			bool noHayRama1 = raiz->rama1 == NULL;
			bool noHayRama2 = raiz->rama2 == NULL;
			bool noHayRama3 = raiz->rama3 == NULL;
			bool noHayRama4 = raiz->rama4 == NULL;
			printf("Este arbol ha sido ramificado\n");
			if (noHayRama1) {
				printf("El cuadrante 1 no tiene cuerpos\n");
			}
			else {
				printf("El cuadrante 1 si tiene cuerpos\n");
				printTree(raiz->rama1);
			}
			if (noHayRama2) {
				printf("El cuadrante 2 no tiene cuerpos\n");
			}
			else {
				printf("El cuadrante 2 si tiene cuerpos\n");
				printTree(raiz->rama2);
			}
			if (noHayRama3) {
				printf("El cuadrante 3 no tiene cuerpos\n");
			}
			else {
				printf("El cuadrante 3 si tiene cuerpos\n");
				printTree(raiz->rama3);
			}
			if (noHayRama4) {
				printf("El cuadrante 4 no tiene cuerpos\n");
			}
			else {
				printf("El cuadrante 4 si tiene cuerpos\n");
				printTree(raiz->rama4);
			}
		}
	}
}
void printPunTree(universo* uni) {
	int nFilas = pow(2, nNiveles);
	for (int i = 0; i < nFilas; i++) {
		for (int j = 0; j < nFilas; j++) {
			if (uni->punTreeMatriz[i][j] != NULL) {
				printTree(uni->punTreeMatriz[i][j]);
			}
			else {
				printf("\n-----------PRINT TREE---------\n");
				printf("La rama en la posicion (%d,%d) esta vacia\n", (i + 1), (j + 1));
			}

		}
	}

}


float centroMasasX(cuaTree* raiz, universo* uni) {
	
	float devuelto = 0;
	float sumatorioMasas = 0;
	cuerpo a;
	cuaTree* ramaActual;

	if (raiz[0].tieneRamas) {

		ramaActual = raiz[0].rama1;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasX * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}

		ramaActual = raiz[0].rama2;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasX * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}

		ramaActual = raiz[0].rama3;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasX * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}

		ramaActual = raiz[0].rama4;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasX * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}
	}
	else {
		int* arrayCuerpos = raiz[0].cuerpos;
		int nCuerpos = raiz[0].nCuerpos;
		int punteroCuerpos;


		for (int i = 0; i < nCuerpos; i++) {
			punteroCuerpos = arrayCuerpos[i];
			a = uni[0].cuerpos[punteroCuerpos];
			sumatorioMasas += a.masa;
			devuelto += a.pos[0] * a.masa;
		}
	}
	
	devuelto = devuelto / sumatorioMasas;

	return devuelto;
}
float centroMasasY(cuaTree* raiz, universo* uni) {

	float devuelto = 0;
	float sumatorioMasas = 0;
	cuerpo a;
	cuaTree* ramaActual;

	if (raiz[0].tieneRamas) {
		
		ramaActual = raiz[0].rama1;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasY * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}

		ramaActual = raiz[0].rama2;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasY * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}

		ramaActual = raiz[0].rama3;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasY * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}

		ramaActual = raiz[0].rama4;
		if (ramaActual != NULL) {
			devuelto += ramaActual->centroMasasY * ramaActual->masaTotal;
			sumatorioMasas += ramaActual->masaTotal;
		}
	}
	else {
		int* arrayCuerpos = raiz[0].cuerpos;
		int nCuerpos = raiz[0].nCuerpos;
		int punteroCuerpos;


		for (int i = 0; i < nCuerpos; i++) {
			punteroCuerpos = arrayCuerpos[i];
			a = uni[0].cuerpos[punteroCuerpos];
			sumatorioMasas += a.masa;
			devuelto += a.pos[1] * a.masa;
		}
	}

	devuelto = devuelto / sumatorioMasas;

	return devuelto;
}
int pathAPunteroFila(int path) {
	int pathMio = path;
	int apoyo = 0;
	int devuelto = 0;
	int comando = 0;
	for (int i = 0; i < nNiveles; i++) {
		comando = pow(10, (nNiveles - (i + 1)));
		apoyo = pathMio / comando;
		switch (apoyo) {
		case 3: devuelto += pow(2, nNiveles - (i + 1)); break;
		case 4: devuelto += pow(2, nNiveles - (i + 1)); break;
		}
		pathMio = pathMio - comando * apoyo;
	}

	return devuelto;

}
int pathAPunteroColumna(int path) {
	int pathMio = path;
	int apoyo = 0;
	int devuelto = 0;
	int comando = 0;
	for (int i = 0; i < nNiveles; i++) {
		comando = pow(10, (nNiveles - (i + 1)));
		apoyo = pathMio / comando;
		switch (apoyo) {
		case 2: devuelto += pow(2, nNiveles - (i + 1)); break;
		case 4: devuelto += pow(2, nNiveles - (i + 1)); break;
		}
		pathMio = pathMio - comando * apoyo;
	}

	return devuelto;

}
int aQueCuadrante(cuerpo a, cuaTree raiz) { 
	//	1	2
	//	3	4
	int cuadrante = 1;

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
void raizTree(universo* uni, cuaTree* raiz) {

	int* cuerpos = (int*)malloc(N * sizeof(int));
	float masaTotal = 0.0;

	for (int i = 0; i < N; i++) {
		cuerpos[i] = i;
		masaTotal += uni[0].cuerpos[i].masa;
	}

	raiz->path = 0;
	raiz->cuadrante = 0;
	raiz->nivelTree = 0;

	raiz->nCuerpos = N;
	raiz->cuerpos = cuerpos;

	raiz->masaTotal = masaTotal;
	raiz->centroMasasX = centroMasasX(raiz, uni);
	raiz->centroMasasY = centroMasasY(raiz, uni);

	raiz->maxX = MAXDIM;
	raiz->maxY = MAXDIM;
	raiz->minX = 0.0 - MAXDIM;
	raiz->minY = 0.0 - MAXDIM;

}
cuaTree* rellenaTree(universo* uni, cuaTree* raiz, int cuadrante, int nCuerpos, 
	int* cuerpos, float masaTotal, float maxX, float maxY, float minX, float minY) {

	//Hago espacio para el cuatree
	struct cuaTree* ramaN = (cuaTree*)malloc(sizeof(struct cuaTree));
	//Le asigno los valores por defecto de un nuevo cuatree. 
	ramaN = new cuaTree;

	//Relleno el cuatree con los datos disponibles (array de cuerpos, numero de cuerpos, path, cuadrante, nivel, masa y limites)
	int* trueCuerpos = (int*)malloc(nCuerpos * sizeof(int));
	for (int i = 0; i < nCuerpos; i++) {
		trueCuerpos[i] = cuerpos[i];
	}
	ramaN->cuerpos = trueCuerpos;
	ramaN->nCuerpos = nCuerpos;
	ramaN->path = raiz[0].path * 10 + cuadrante;
	ramaN->cuadrante = cuadrante;
	ramaN->nivelTree = raiz[0].nivelTree + 1;
	ramaN->masaTotal = masaTotal;
	ramaN->maxX = maxX;
	ramaN->minX = minX;
	ramaN->maxY = maxY;
	ramaN->minY = minY;
	ramaN->padre = raiz;

	if (ramaN->nivelTree==nNiveles) {
		ramaN->idFila = pathAPunteroFila(ramaN->path);
		ramaN->idColumna = pathAPunteroColumna(ramaN->path);
		uni->punTreeMatriz[ramaN->idFila][ramaN->idColumna] = ramaN;
	}
	
	return ramaN;

}
void ramificaTree(cuaTree* raiz, universo* uni) {
	//VariablesDeApoyo
	//printf("Nivel: %d	Path: %d\n", raiz[0].nivelTree, raiz[0].path);
	raiz[0].tieneRamas = true;
	cuaTree* rama;
	bool ramificarMas = (raiz[0].nivelTree + 1 < nNiveles);//Es el nivel que creo ahora el nivel más bajo?

	float minX = raiz[0].minX;
	float maxX = raiz[0].maxX;
	float minY = raiz[0].minY;
	float maxY = raiz[0].maxY;
	float midX = (minX + maxX) / 2;
	float midY = (minY + maxY) / 2;

	int nCuerposTotales = raiz[0].nCuerpos;

	int nCuerposRama1 = 0;
	int nCuerposRama2 = 0;
	int nCuerposRama3 = 0;
	int nCuerposRama4 = 0;

	int* cuerpos1 = (int*)malloc(nCuerposTotales * sizeof(int));
	int* cuerpos2 = (int*)malloc(nCuerposTotales * sizeof(int));
	int* cuerpos3 = (int*)malloc(nCuerposTotales * sizeof(int));
	int* cuerpos4 = (int*)malloc(nCuerposTotales * sizeof(int));

	float masaTotal1 = 0;
	float masaTotal2 = 0;
	float masaTotal3 = 0;
	float masaTotal4 = 0;

	int numCuerpo;
	int num;
	cuerpo cuerpoTree;
	//Recorrer cuerpos del tree

	for (int i = 0; i < nCuerposTotales; i++) {
		numCuerpo = raiz[0].cuerpos[i];
		cuerpoTree = uni[0].cuerpos[numCuerpo];
		num = aQueCuadrante(cuerpoTree, raiz[0]);

		switch (num) {

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

		case 4:	cuerpos4[nCuerposRama4] = numCuerpo;
			nCuerposRama4++;
			masaTotal4 += cuerpoTree.masa;
			break;
		}

	}




	//Rama 1
	if (nCuerposRama1 > 0) {
		//Si tiene cuerpos, se crea y rellena la rama. 
		rama = rellenaTree(uni, raiz, 1, nCuerposRama1, cuerpos1, masaTotal1, midX, maxY, minX, midY);
		raiz[0].rama1 = rama;
		//Si no es un nivel suficiente, se ramifica más
		if (ramificarMas) {
			ramificaTree(rama, uni);
		}

		//Se obtiene el centro de masas
		rama->centroMasasX = centroMasasX(rama, uni);
		rama->centroMasasY = centroMasasY(rama, uni);

	}
	free(cuerpos1);


	//Rama 2  
	if (nCuerposRama2 > 0) {
		//Si tiene cuerpos, se crea y rellena la rama. 
		rama = rellenaTree(uni, raiz, 2, nCuerposRama2, cuerpos2, masaTotal2, maxX, maxY, midX, midY);
		raiz[0].rama2 = rama;
		//Si no es un nivel suficiente, se ramifica más
		if (ramificarMas) {
			ramificaTree(rama, uni);
		}

		//Se obtiene el centro de masas
		rama->centroMasasX = centroMasasX(rama, uni);
		rama->centroMasasY = centroMasasY(rama, uni);

	}
	free(cuerpos2);

	//Rama 3 
	if (nCuerposRama3 > 0) {
		//Si tiene cuerpos, se crea y rellena la rama. 
		rama = rellenaTree(uni, raiz, 3, nCuerposRama3, cuerpos3, masaTotal3, midX, midY, minX, minY);
		raiz[0].rama3 = rama;
		//Si no es un nivel suficiente, se ramifica más
		if (ramificarMas) {
			ramificaTree(rama, uni);
		}

		//Se obtiene el centro de masas
		rama->centroMasasX = centroMasasX(rama, uni);
		rama->centroMasasY = centroMasasY(rama, uni);

	}
	free(cuerpos3);

	//Rama 4 
	if (nCuerposRama4 > 0) {
		//Si tiene cuerpos, se crea y rellena la rama. 
		rama = rellenaTree(uni, raiz, 4, nCuerposRama4, cuerpos4, masaTotal4, maxX, midY, midX, minY);
		raiz[0].rama4 = rama;
		//Si no es un nivel suficiente, se ramifica más
		if (ramificarMas) {
			ramificaTree(rama, uni);
		}

		//Se obtiene el centro de masas
		rama->centroMasasX = centroMasasX(rama, uni);
		rama->centroMasasY = centroMasasY(rama, uni);

	}
	free(cuerpos4);

}
void cleanMatrizTree(universo* uni) {
	//punTree
	int nFilas = (int)pow(2, nNiveles) + 1e-9;
	int nColumnas = (int)pow(2, nNiveles) + 1e-9;
	for (int i = 0; i < nFilas; i++) {
		for (int j = 0; j < nColumnas; j++) {
			uni->punTreeMatriz[i][j] = NULL;
		}
	}
}
void liberaTree(cuaTree* raiz) {
	free(raiz->cuerpos);
	raiz->nCuerpos = 0;
	if (raiz->rama1 != NULL) {//Este arbol tenia un hijo en rama 1. 
		liberaTree(raiz->rama1); //Se invoca esta funcion para liberar el espacio de los punteros de su hijo
		free(raiz->rama1); //liberas al propio hijo
		raiz->rama1 = NULL;
	}
	if (raiz->rama2 != NULL) { //Lo mismo con la rama 2, 3 y 4
		liberaTree(raiz->rama2);
		free(raiz->rama2);
		raiz->rama2 = NULL;
	}
	if (raiz->rama3 != NULL) {
		liberaTree(raiz->rama3);
		free(raiz->rama3);
		raiz->rama3 = NULL;
	}
	if (raiz->rama4 != NULL) {
		liberaTree(raiz->rama4);
		free(raiz->rama4);
		raiz->rama4 = NULL;
	}
	raiz->tieneRamas = false;
}



void calculoFuerzaCuerpoCuerpo(universo* uni, int idCuerpo1, int idCuerpo2) {
	//Force iterate toma los ids de los cuerpos y los calcula los unos con los otros
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
void calculoFuerzaCuerpoTree(universo* uni, int idCuerpo, cuaTree* rama) {
	//Coge el cuerpo idCuerpo, y el arbol rama. 
	cuerpo cuerpo = uni->cuerpos[idCuerpo];
	float posX1 = cuerpo.pos[0];
	float posY1 = cuerpo.pos[1];
	float posX2 = rama->centroMasasX;
	float posY2 = rama->centroMasasY;

	float M1 = cuerpo.masa;
	float M2 = rama->masaTotal;

	float difX = posX1 - posX2;
	float difY = posY1 - posY2;

	float disTotal = sqrt(difX * difX + difY * difY);

	float F = G * M1 * M2 / (disTotal * disTotal);

	float cos = difX / disTotal;
	float sen = difY / disTotal;

	float Fx = F * cos;
	float Fy = F * sen;
	//Hace literalmente lo mismo que en NCuerposCPU pero tomando los datos de rama
	cuerpo.fuerzas[0] -= Fx;
	cuerpo.fuerzas[1] -= Fy;

	uni->cuerpos[idCuerpo] = cuerpo;
	//Mete el nuevo cuerpo en uni
}


void forceIterateTree(universo* uni, int idFila, int idColumna) {
	//Comprobar que el cuaTree idFila, idColumna existe
	struct cuaTree* puntTree = uni->punTreeMatriz[idFila][idColumna];
	struct cuaTree* puntTree2;
	if (puntTree == NULL) {
		//No hay cuerpos en el rango de ese tree
	}
	else {
		//Hay punteros en el rango de ese tree. 
			//Calculo las fuerzas de los cuerpos consigo mismos
			//Tiene un numero de cuerpos >=1
		int nCuerpos = puntTree->nCuerpos;
		for (int i = 0; i < nCuerpos; i++) {
			for (int j = i + 1; j < nCuerpos; j++) {
				calculoFuerzaCuerpoCuerpo(uni, puntTree->cuerpos[i], puntTree->cuerpos[j]);
				//Los cuerpos emiten fuerza entre si con sus posiciones verdaderas
			}
		}

		//Calculo las fuerzas con el resto de Trees
		//Numero de filas y columnas en la matriz de trees
		int nFilas = (int)pow(2, nNiveles) + 1e-9;
		int nColumnas = (int)pow(2, nNiveles) + 1e-9;
		int idCuerpo = -1;
		for (int i = 0; i < nFilas; i++) {
			for (int j = 0; j < nColumnas; j++) {
				//Cojo el tree de posicion i,j
				puntTree2 = uni->punTreeMatriz[i][j];
				if (puntTree2 != NULL) {
					//Si el tree i,j no es un null, tiene masatotal y centros de masa
					if (i == idFila && j == idColumna) {
						//Si es este tree no lo comparo consigo mismo
					}
					else {
						//Comparo cada cuerpo con el centro de masas y su masa total
						for (int k = 0; k < nCuerpos; k++) {
							idCuerpo = puntTree->cuerpos[k];
							calculoFuerzaCuerpoTree(uni, idCuerpo, puntTree2);
							//En esencia, a cada cuerpo del tree (idFila, idColumna) se le hace la iteracion
							//de fuerza con el tree(i,j)
						}
					}
				}
			}
		}
	}

}
void newForcesTree(universo* uni) {
	//Las fuerzas pasan a ser 0
	for (int i = 0; i < N; i++) {
		uni->cuerpos[i].fuerzas[0] = 0;
		uni->cuerpos[i].fuerzas[1] = 0;
	}
	//Se calculan las nuevas Fuerzas, yendo de arbol en arbol
	int nFilas = (int)pow(2, nNiveles) + 1e-9;
	int nColumnas = (int)pow(2, nNiveles) + 1e-9;
	for (int i = 0; i < nFilas; i++) {
		for (int j = 0; j < nColumnas; j++) {
			forceIterateTree(uni, i, j);
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
void iterateUniverseTreeCPU(universo* uni, int nSegundos, bool print) {
	int timeLeft = nSegundos;
	int nIteration = 0;
	int nIteracionesTotales = nSegundos / TIMELAPSE;
	int countTree = 0;

	struct cuaTree* raiz = (cuaTree*)malloc(sizeof(cuaTree));
	raiz = new cuaTree;
	raizTree(uni, raiz);
	ramificaTree(raiz, uni);
	
	while (timeLeft >= TIMELAPSE) {
		if (print) {
			writeData(uni, nIteration, nIteracionesTotales + 1);
		}
		if (countTree == CLEANTREEITERATION) {
			liberaTree(raiz);
			cleanMatrizTree(uni);
			raizTree(uni, raiz);
			ramificaTree(raiz, uni);
			countTree = 0;
		}
		newForcesTree(uni);
		newAcel(uni);
		newPosition(uni);
		newSpeed(uni);
		timeLeft -= TIMELAPSE;
		nIteration++;
		countTree++;
	}
	if (print) {
		writeData(uni, nIteration, nIteracionesTotales + 1);
	}
}

int main() {

	clock_t tiempo_inicio, tiempo_final;
	double segundos;
	int tiempoIteracion = 36000;

	struct universo* uni = (universo*)malloc(sizeof(universo));
	uni = new universo;
	crearUniversoAleatorio(uni); //Rellena uni
	cleanMatrizTree(uni);//Limpia la matriz de punteros a ramas finales de uni
	
	//printCuerpos(uni, 0, true, true);
	printf("Comienzo de la iteracion del universo\n");
	printf("	Numero de cuerpos:		%d\n", N);
	printf("	Segundos por iteracion:		%d\n", TIMELAPSE);
	printf("	Tiempo a iterar:		%d\n", tiempoIteracion);
	printf("	Numero de iteraciones:		%d\n", tiempoIteracion / TIMELAPSE);
	
	
	tiempo_inicio = clock();
	iterateUniverseTreeCPU(uni, tiempoIteracion, true);
	tiempo_final = clock();

	segundos = (double)(tiempo_final - tiempo_inicio) / CLOCKS_PER_SEC; /*según que estes midiendo el tiempo en segundos es demasiado grande*/

	printf("\nTIEMPO TARDADO: %f\n", segundos);




	return 0;
}
