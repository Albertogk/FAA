from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import norm, logistic
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

        return n_err / len(datos)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self, particionado, dataset, diccionario,
                   atributosDiscretos, alpha=1, seed=None):
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
            pred = self.clasifica(datostest, atributosDiscretos, diccionario,
                                  alpha=alpha)

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

        if datostrain is None:
            return

        n_atributos = len(datostrain[0]) - 1
        n_clases = len(diccionario[-1])

        ocurrencias = []

        medias = []
        desv = []

        # Lista donde se guarda el número de veces que aparece cada clase
        ocurrencias_clase = [0] * n_clases

        # Inicialuizacion de listas de datos
        for i in range(n_atributos):

            ocurrencias.append([])

            if atributosDiscretos[i]:
                # Para atributos discretos se crea una lista de listas donde
                # se guarda el número de ocurrencias para cada clase:
                # [a1: [v1: [n_c1, n_c2...], v2: [[n_c1, n_c2...]...], a2: ...]
                n_valores = len(diccionario[i])
                for j in range(n_valores):
                    ocurrencias[i].append([0] * n_clases)

                medias.append([])
                desv.append([])

            else:
                # Para atributos continuos se crean dos listas: media y desviacion tipica
                # Se guarda un valor para cada atributo y clase:
                # [a1: [media_c1, media_c2...], a2: [media_c1...], ...]

                medias.append([0] * n_clases)
                desv.append([0] * n_clases)

        for fila in datostrain:

            # Se actualiza el número de elementos que contiene cada clase
            ocurrencias_clase[int(fila[-1])] += 1

            # Para cada atributo:
            for i in range(len(fila) - 1):

                # Si el atributo es discreto se aumenta en uno la posicion
                # correspondiente
                # i : atributo
                # fila[i] : valor
                # fila[-1] : clase
                if atributosDiscretos[i]:
                    ocurrencias[i][fila[i]][int(fila[-1])] += 1
                else:
                    # Si es continuo, se suma el valor en el array de medias
                    medias[i][int(fila[-1])] += fila[i]

        # Se calculan las medias usando la suma obtenida y el número de elementos
        # para cada clase
        for i in range(len(medias)):
            if not atributosDiscretos[i]:
                for j in range(len(medias[i])):
                    medias[i][j] /= ocurrencias_clase[j]

        # Se calcula la desviacion media
        for fila in datostrain:
            for i in range(len(fila) - 1):
                if not atributosDiscretos[i]:
                    desv[i][int(fila[-1])] += (fila[i] - medias[i][
                        int(fila[-1])]) ** 2

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
            probabilidades.append([1] * self.n_clases)

        # Lista que guarda para que atributos será necesario aplicar Laplace
        laplace = [0] * (len(datostest[0]) - 1)

        for fila in datostest:
            for i in range(len(fila) - 1):
                if atributosDiscretos[i]:
                    for j in range(self.n_clases):
                        if self.ocurrencias[i][fila[i]][j] == 0:
                            laplace[i] = alpha

        # Calculo de probabilidades
        idx_fila = 0

        for fila in datostest:
            for j in range(self.n_clases):
                for i in range(len(fila) - 1):

                    # Si el atributo es discreto, se calcula la probabilidad utilizando
                    # "ocurrencias" de entrenamiento
                    # Con el metodo aproximado de Naive-Bayes
                    if atributosDiscretos[i]:
                        n_valores = len(diccionario[i])
                        probabilidades[idx_fila][j] *= (self.ocurrencias[i][
                                                            fila[i]][j] +
                                                        laplace[i]) / (
                                                                   self.ocurrencias_clase[
                                                                       j] +
                                                                   laplace[
                                                                       i] * n_valores)

                    # Para atributos continuos se calcula la probabilidad usando la distibucion normal y los datos de medias y desviaciones tipicas
                    else:
                        probabilidades[idx_fila][j] *= norm.pdf(fila[i],
                                                                self.medias[i][
                                                                    j],
                                                                self.desv[i][
                                                                    j])

                # En ambos casos, después de procesar los atributos, se multiplica la probabilidad por el prior
                probabilidades[idx_fila][j] *= (
                            self.ocurrencias_clase[j] / len(datostest))

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
        self.norm = True
        self.probabilidades = None
        self.VI = None

    def calcularMediasDesv(self, datos, nominalAtributos):

        n_filas, n_atributos = datos.shape
        medias = np.zeros(n_atributos)
        desv = np.zeros(n_atributos)

        for j in range(n_atributos):
            for i in range(n_filas):
                if not nominalAtributos[j]:
                    medias[j] += datos[i][j]
            medias[j] /= n_filas

        for j in range(n_atributos):
            for i in range(n_filas):
                if not nominalAtributos[j]:
                    desv[j] += (datos[i][j] - medias[j]) ** 2

            desv[j] = np.sqrt(desv[j]) / n_filas

        return medias, desv

    def normalizarDatos(self, datos, nominalAtributos):

        datos_normalizados = np.empty(datos.shape)

        n_filas, n_atributos = datos.shape
        for i in range(n_filas):
            for j in range(n_atributos):
                if not nominalAtributos[j]:
                    datos_normalizados[i][j] = (datos[i][j] - self.medias[j]) / \
                                               self.desv[j]
                else:
                    datos_normalizados[i][j] = datos[i][j]

        return datos_normalizados

    def dist_euclidea(self, x1, x2):
        return np.sqrt(sum((x1 - x2) * (x1 - x2)))

    def dist_manhattan(self, x1, x2):
        return sum(np.abs(x1 - x2))

    def dist_mahalanobis(self, x1, x2, VI):
        res = np.dot(np.dot((x1 - x2).T, VI), (x1 - x2))
        return np.sqrt(res)

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario,
                      norm=True):
        self.norm = norm
        self.medias, self.desv = self.calcularMediasDesv(datosTrain,
                                                         atributosDiscretos)
        if self.norm:
            self.datos_train_norm = self.normalizarDatos(datosTrain,
                                                         atributosDiscretos)
        else:
            self.datos_train_norm = datosTrain

        self.VI = np.linalg.inv(np.cov(self.datos_train_norm[:, :-1].T))

    def clasifica(self, datostest, atributosDiscretos, diccionario,
                  distancia="euclidea", k=3):

        distancias = np.zeros((len(datostest), len(self.datos_train_norm), 2))
        self.probabilidades = np.zeros(datostest.shape[0])

        if self.norm:
            datos_test_norm = self.normalizarDatos(datostest,
                                                   atributosDiscretos)
        else:
            datos_test_norm = datostest

        for i in range(datos_test_norm.shape[0]):
            for j in range(self.datos_train_norm.shape[0]):
                clase_j = self.datos_train_norm[j][-1]

                if distancia == "euclidea":
                    distancias[i][j] = [
                        self.dist_euclidea(datos_test_norm[i][:-1],
                                           self.datos_train_norm[j][:-1]),
                        clase_j]
                elif distancia == "manhattan":
                    distancias[i][j] = [
                        self.dist_manhattan(datos_test_norm[i][:-1],
                                            self.datos_train_norm[j][:-1]),
                        clase_j]
                elif distancia == "mahalanobis":

                    distancias[i][j] = [
                        self.dist_mahalanobis(datos_test_norm[i][:-1],
                                              self.datos_train_norm[j][:-1],
                                              self.VI), clase_j]

        pred = np.zeros(datostest.shape[0])

        for i in range(datos_test_norm.shape[0]):

            clases = np.zeros(len(diccionario[-1]))
            sort_dist = sorted(distancias[i], key=lambda x: x[0])[0:k]

            for e in sort_dist:
                clases[int(e[1])] += 1

            self.probabilidades[i] = clases[1] / sum(clases)

            pred[i] = np.argmax(clases)

        return pred


class ClasificadorRegresionLogistica(Clasificador):

    def __init__(self, alpha=0.001, n_epocas=100):
        self.alpha = alpha
        self.n_epocas = n_epocas
        self.w = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        w = np.random.rand(datosTrain.shape[1]) - 0.5
        for epoca in range(self.n_epocas):
            for i in range(datosTrain.shape[0]):
                x_i = np.concatenate(([1], datosTrain[i][:-1]))
                t_i = datosTrain[i][-1]
                sigma = self.sigmoid(np.inner(w, x_i))
                w = w - self.alpha * (sigma - t_i) * x_i

        self.w = w

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pred = np.empty(datosTest.shape[0])
        for idx in range(datosTest.shape[0]):
            x = datosTest[idx]
            if np.inner(self.w, np.concatenate(([1], x[:-1]))) < 0:
                pred[idx] = 0
            else:
                pred[idx] = 1

        return pred


class AlgoritmoGenetico(Clasificador):

    def __init__(self):
        self.ultimo_individuo = None

    def transformar_datos(self, datos, diccionario):

        datos_transformados = np.zeros(
            (datos.shape[0], sum(map(len, diccionario[:-1])) + 1), dtype=int)

        for i in range(datos.shape[0]):
            cnt = 0
            for j in range(datos.shape[1] - 1):
                datos_transformados[i][cnt + int(datos[i][j])] = 1
                cnt += len(diccionario[j])

            datos_transformados[i][-1] = datos[i][-1]
        return datos_transformados

    def get_aciertos(self, reglas, individuo, diccionario):

        aciertos = 0
        for i in range(len(reglas)):
            for k in range(individuo.shape[0]):
                cnt = 0
                for j in range(len(diccionario[:-1])):
                    next_cnt = cnt + len(diccionario[j])
                    aux = reglas[i, cnt:next_cnt] & individuo[k][cnt:next_cnt]

                    if 1 not in aux:
                        break

                    cnt = next_cnt

                if cnt == len(individuo[k][:-1]):
                    aciertos += (individuo[k][-1] == reglas[i][-1])
                else:
                    aciertos += ((individuo[k][-1] + 1) % 2 == reglas[i][-1])

        return aciertos / (reglas.shape[0]*individuo.shape[0])

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario,
                      elitismo=0.05, tamanio_poblacion=50, n_epocas=100,
                      puntos_cruce=1, p_mutacion=0.01, reglas_por_ind=3):

        if reglas_por_ind > datosTrain.shape[0]/tamanio_poblacion:
            reglas_por_ind = int(datosTrain.shape[0]/tamanio_poblacion)
            print("Overflow de reglas_por ind. Valor establecido en",
                  reglas_por_ind)

        base_reglas = self.transformar_datos(datosTrain, diccionario)
        elites = int(tamanio_poblacion * elitismo)

        if elites % 2 != tamanio_poblacion % 2:
            elites += 1

        # Generacion de poblacion inicial
        generacion = np.empty(
            (tamanio_poblacion, reglas_por_ind, base_reglas.shape[1]),
            dtype=int)

        for i in range(generacion.shape[0]):
            for j in range(reglas_por_ind):
                generacion[i][j] =\
                    base_reglas[np.random.randint(0, high=base_reglas.shape[0])]

        next_generacion = np.empty(
            (tamanio_poblacion, reglas_por_ind, base_reglas.shape[1]),
            dtype=int)

        for epoca in range(n_epocas):

            fitness = np.empty(generacion.shape[0])
            for i in range(fitness.shape[0]):
                fitness[i] = self.get_aciertos(base_reglas, generacion[i],
                                               diccionario)

            # elitismo
            if elites > 0:
                ind = np.argpartition(fitness, -elites)[-elites:]
                next_generacion[:elites] = generacion[ind]

            # ruleta
            fitness_ruleta = fitness / sum(fitness)

            for i in range(elites, tamanio_poblacion, 2):
                progenitores = np.empty(2, dtype=int)
                for d in range(progenitores.shape[0]):
                    numero = np.random.rand()
                    total = 0
                    for k in range(len(fitness_ruleta)):
                        total += fitness_ruleta[k]
                        if numero <= total:
                            progenitores[d] = k
                            break

                # cruce uniforme
                if puntos_cruce is None:
                    for k in range(reglas_por_ind):
                        for bit in range(generacion.shape[2]):
                            if np.random.random() < 0.5:
                                next_generacion[i][k][bit] = generacion[progenitores[0]][k][bit]
                                next_generacion[i+1][k][bit] = generacion[progenitores[1]][k][bit]

                            else:
                                next_generacion[i][k][bit] = generacion[progenitores[1]][k][bit]
                                next_generacion[i + 1][k][bit] = generacion[progenitores[0]][k][bit]

                # cruce en n puntos
                else:
                    permutation = np.random.permutation(np.arange(1, generacion.shape[2]))
                    permutation = sorted(permutation[:puntos_cruce])

                    permutation = np.concatenate((permutation, [generacion.shape[2]]))

                    for k in range(reglas_por_ind):
                        cnt = 0
                        for n in range(permutation.shape[0]):
                            if n % 2 == 0:
                                next_generacion[i][k][cnt:permutation[n]] = generacion[progenitores[0]][k][cnt:permutation[n]]
                                next_generacion[i + 1][k][cnt:permutation[n]] = generacion[progenitores[1]][k][cnt:permutation[n]]
                            else:
                                next_generacion[i][k][cnt:permutation[n]] = generacion[progenitores[1]][k][cnt:permutation[n]]
                                next_generacion[i + 1][k][cnt:permutation[n]] = generacion[progenitores[0]][k][cnt:permutation[n]]

                            cnt = permutation[n]

            # mutacion
            if p_mutacion > 0:
                for i in range(next_generacion.shape[0]):
                    for k in range(reglas_por_ind):
                        for bit in range(next_generacion.shape[2]):
                            if np.random.random() < p_mutacion:
                                next_generacion[i][k][bit] = (next_generacion[i][k][bit] + 1)%2

            fitness_medio = np.mean(fitness)
            fitness_max = max(fitness)
            print("Generacion:", epoca)
            print("Fitness medio:", fitness_medio)
            print("Fitness mejor individuo:", fitness_max)
            print('###################################################')

            generacion = next_generacion

        fitness = np.empty(generacion.shape[0])
        for i in range(fitness.shape[0]):
            fitness[i] = self.get_aciertos(base_reglas, generacion[i],
                                           diccionario)

        fitness_medio = np.mean(fitness)
        fitness_max = max(fitness)
        print("Generacion:", n_epocas)
        print("Fitness medio:", fitness_medio)
        print("Fitness mejor individuo:", fitness_max)
        print('###################################################')

        ind = np.argpartition(fitness, -1)[-1:]
        self.ultimo_individuo = generacion[ind[0]]


    def obtener_clase(self, dato, n_valores, n_atributos):

        votos = np.zeros(2, dtype=int)
        for i in range(self.ultimo_individuo.shape[0]):
            cnt = 0
            cumple_regla = True
            for j in range(n_atributos-1):
                next_cnt = cnt + n_valores[j]
                aux = dato[cnt:next_cnt] & self.ultimo_individuo[i][cnt:next_cnt]

                if 1 not in aux:
                    cumple_regla = False
                    break

                cnt = next_cnt

            if cumple_regla:
                votos[self.ultimo_individuo[i][-1]] += 1
            else:
                votos[(self.ultimo_individuo[i][-1]+1)%2] += 1

        if votos[0] > votos[1]:
            return 0
        else:
            return 1


    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pred = np.empty(datosTest.shape[0])
        n_valores = list(map(len, diccionario[:-1]))
        n_atributos = len(diccionario)
        datosTestTrain = self.transformar_datos(datosTest, diccionario)

        for idx in range(datosTest.shape[0]):
            pred[idx] = self.obtener_clase(datosTestTrain[idx], n_valores, n_atributos)

        return pred
