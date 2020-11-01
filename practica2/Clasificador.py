from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import euclidean, cityblock, mahalanobis

class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta


    @abstractmethod
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    def error(self, datos, pred):
        n_err = 0

        for i in range(len(datos)):
            if datos[i][-1] != pred[i]:
                n_err += 1

        return n_err/len(datos)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self, particionado, dataset, diccionario, atributosDiscretos, alpha = 1, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opcion es repetir la validación simple un número especificado de veces,
        # obteniendo en cada una un error. Finalmente se calcularia la media.

        n_datos = len(dataset.datos)
        particiones = particionado.creaParticiones(n_datos, seed)

        error = []
        for particion in particiones:
            datostest = dataset.extraeDatos(particion.indicesTest)
            datostrain = dataset.extraeDatos(particion.indicesTrain)

            self.entrenamiento(datostrain, atributosDiscretos, diccionario)
            pred = self.clasifica(datostest, atributosDiscretos, diccionario, alpha=alpha)

            error.append(self.error(datostest, pred))

        return error


class ClasificadorNaiveBayes(Clasificador):

    def __init__(self):
        self.ocurrencias = []
        self.ocurrencias_clase = []

        self.medias = []
        self.desv = []

        self.n_clases = 0
        self.numero_atributos = 0

        self.probabilidades = []

        self.prediccion = []

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        if not datostrain:
            return

        n_atributos = len(datostrain[0]) - 1
        n_clases = len(diccionario[-1])

        ocurrencias = []

        medias = []
        desv = []

        # Lista donde se guarda el número de veces que aparece cada clase
        ocurrencias_clase = [0]*n_clases

        # Inicialuizacion de listas de datos
        for i in range(n_atributos):

            ocurrencias.append([])

            if atributosDiscretos[i]:
                # Para atributos discretos se crea una lista de listas donde
                # se guarda el número de ocurrencias para cada clase:
                # [a1: [v1: [n_c1, n_c2...], v2: [[n_c1, n_c2...]...], a2: ...]
                n_valores = len(diccionario[i])
                for j in range(n_valores):
                    ocurrencias[i].append([0]*n_clases)

                medias.append([])
                desv.append([])

            else:
                # Para atributos continuos se crean dos listas: media y desviacion tipica
                # Se guarda un valor para cada atributo y clase:
                # [a1: [media_c1, media_c2...], a2: [media_c1...], ...]

                medias.append([0]*n_clases)
                desv.append([0]*n_clases)

        for fila in datostrain:

            # Se actualiza el número de elementos que contiene cada clase
            ocurrencias_clase[fila[-1]] += 1

            # Para cada atributo:
            for i in range(len(fila)-1):

                # Si el atributo es discreto se aumenta en uno la posicion
                # correspondiente
                # i : atributo
                # fila[i] : valor
                # fila[-1] : clase
                if atributosDiscretos[i]:
                    ocurrencias[i][fila[i]][fila[-1]] += 1
                else:
                    # Si es continuo, se suma el valor en el array de medias
                    medias[i][fila[-1]] += fila[i]


        # Se calculan las medias usando la suma obtenida y el número de elementos
        # para cada clase
        for i in range(len(medias)):
            if not atributosDiscretos[i]:
                for j in range(len(medias[i])):
                    medias[i][j] /= ocurrencias_clase[j]

        # Se calcula la desviacion media
        for fila in datostrain:
            for i in range(len(fila)-1):
                if not atributosDiscretos[i]:
                    desv[i][fila[-1]] += (fila[i] - medias[i][fila[-1]])**2

        for i in range(len(desv)):
            if not atributosDiscretos[i]:
                for j in range(len(desv[i])):
                    desv[i][j] /= ocurrencias_clase[j]
                    desv[i][j] = np.sqrt(desv[i][j])

        # Se actualizan los atributos de la clase con datos relevantes
        self.ocurrencias = ocurrencias
        self.ocurrencias_clase = ocurrencias_clase

        self.medias = medias
        self.desv = desv

        self.n_clases = n_clases
        self.numero_atributos = n_atributos

        return

    def clasifica(self, datostest, atributosDiscretos, diccionario, alpha=1):

        pred = []

        # Inicializacion de la lista que almacenara las probabilidades de cada elemento para cada clase
        probabilidades = []

        for _ in datostest:
            probabilidades.append([1]*self.n_clases)

        # Lista que guarda para que atributos será necesario aplicar Laplace
        laplace = [0]*(len(datostest[0])-1)

        for fila in datostest:
            for i in range(len(fila)-1):
                if atributosDiscretos[i]:
                    for j in range(self.n_clases):
                        if self.ocurrencias[i][fila[i]][j] == 0:
                            laplace[i] = alpha

        # Calculo de probabilidades
        idx_fila = 0

        for fila in datostest:
            for j in range(self.n_clases):
                for i in range(len(fila)-1):

                    # Si el atributo es discreto, se calcula la probabilidad utilizando
                    # "ocurrencias" de entrenamiento
                    # Con el metodo aproximado de Naive-Bayes
                    if atributosDiscretos[i]:
                        n_valores = len(diccionario[i])
                        probabilidades[idx_fila][j] *= (self.ocurrencias[i][fila[i]][j] + laplace[i])/(self.ocurrencias_clase[j] + laplace[i]*n_valores)

                    # Para atributos continuos se calcula la probabilidad usando la distibucion normal y los datos de medias y desviaciones tipicas
                    else:
                        probabilidades[idx_fila][j] *= norm.pdf(fila[i], self.medias[i][j], self.desv[i][j])

                # En ambos casos, después de procesar los atributos, se multiplica la probabilidad por el prior
                probabilidades[idx_fila][j] *= (self.ocurrencias_clase[j]/len(datostest))

            # Se obtiene la probabilidad diviendiendo entre la suma
            total = sum(probabilidades[idx_fila])
            for j in range(self.n_clases):
                probabilidades[idx_fila][j] /= total

            idx_fila += 1

        self.probabilidades = probabilidades

        # Se almacena en pred la clase que se predice
        for i in range(len(datostest)):
            cl = probabilidades[i].index(max(probabilidades[i]))
            pred.append(cl)

        self.prediccion = pred

        return pred


class ClasificadorVecinosProximos(Clasificador):

    def __init__(self):
        self.datos_train_norm = None
        self.medias = None
        self.desv = None

    def calcularMediasDesv(self, datos, nominalAtributos):

        n_atributos = datos.shape[1]
        medias = np.zeros(n_atributos)
        desv = np.zeros(n_atributos)

        for i in range(n_atributos):
            if not nominalAtributos[i]:
                medias[i] = np.mean(datos[:, i])
                desv[i] = np.std(datos[:, i])

        return medias, desv

    def normalizarDatos(self, datos, nominalAtributos):

        datos_normalizados = np.empty(datos.shape)

        n_filas, n_atributos = datos.shape
        for i in range(n_filas):
            for j in range(n_atributos):
                if not nominalAtributos[j]:
                    datos_normalizados[i][j] = (datos[i][j] - self.medias[j])/self.desv[j]
                else:
                    datos_normalizados[i][j] = datos[i][j]

        return datos_normalizados


    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        self.medias, self.desv = self.calcularMediasDesv(datosTrain, atributosDiscretos)
        self.datos_train_norm = self.normalizarDatos(datosTrain, atributosDiscretos)


    def clasifica(self, datostest, atributosDiscretos, diccionario, distancia=euclidean, k=3):

        distancias = np.zeros((len(datostest), len(self.datos_train_norm)))
        datos_test_norm = self.normalizarDatos(datostest, atributosDiscretos)

        for i in range(datos_test_norm.shape[0]):
            for j in range(self.datos_train_norm.shape[0]):
                distancias[i][j] = (distancia(datos_test_norm[i], self.datos_train_norm[j]), j)

        pred = np.zeros(datostest.shape[0])

        for i in range(datos_test_norm.shape[0]):
            clases = np.zeros(len(diccionario[-1]))

            sort_dist = np.sort(distancias[i])[0:k]
            for e in sort_dist:
                clases[e[1]] += 1
            pred[i] = np.argmax(clases)

        return pred




