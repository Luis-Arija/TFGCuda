# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
nombre_archivo = 'archivo.txt'
counter = 0;
texto = "";
# Abrir el archivo en modo lectura ('r')
with open(nombre_archivo, 'r') as archivo:
    # Iterar sobre cada línea del archivo
    for linea in archivo:
        # Dividir la línea en palabras
        numeros = linea.split(';');
        # Intentar convertir cada palabra a un número
        if (counter == 0):

            informacion = [[(0.0, 0.0) for j in range(int(numeros[1]))] for i in range(int(numeros[0]))]
            #informacion = np.zeros((int(numeros[0]), int(numeros[1])));
            #for lista_de_tuplas in informacion:
                #print(lista_de_tuplas)


        else: 
           tupla = (float(numeros[2]), float(numeros[3]));
           informacion[int(numeros[0])][int(numeros[1])] = tupla;
        counter=counter+1;

#print ("-----------------------------------")
#for lista_de_tuplas in informacion:
#    print(lista_de_tuplas)

#print ("-----------------------------------")
#print (informacion);


# Inicializa el gráfico
fig, ax = plt.subplots()
sc = ax.scatter([], [], marker='o', color='b', s = 20);

# Ajusta los límites del gráfico
ax.set_xlim(-50000, 50000)  # Ajusta según sea necesario
ax.set_ylim(-50000, 50000)  # Ajusta según sea necesario

# Función de inicialización
def init():
    sc.set_offsets([(0,0)])
    return sc,

# Función de actualización en cada cuadro de la animación
def update(frame):
    datos = informacion[frame]
    x, y = zip(*datos)
    sc.set_offsets(list(zip(x, y)))
    return sc,

# Crea la animación
num_frames = len(informacion)
animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

# Muestra la animación
plt.show()