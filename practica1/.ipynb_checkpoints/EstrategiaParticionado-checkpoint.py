from abc import ABCMeta, abstractmethod
import random

class Particion():

    # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones
    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []


#####################################################################################################

class EstrategiaParticionado:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, n_datos, seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    def __init__(self, porcentaje_test, n_ejecuciones):
        self.porcentaje_test = porcentaje_test
        self.n_ejecuciones = n_ejecuciones


    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el número de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, n_datos, seed=None):
        random.seed(seed)
        ids = list(range(0, n_datos))
        n_test = round(n_datos*self.porcentaje_test)
        particiones = []

        for i in range(0, self.n_ejecuciones):
            random.shuffle(ids)
            particion = Particion()
            particion.indicesTest = ids[0:n_test-1]
            particion.indicesTrain = ids[n_test:]
            particiones.append(particion)

        return particiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, k):
        self.k = k

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, n_datos, seed=None):
        datos_por_fold = n_datos//self.k
        print(datos_por_fold)
        random.seed(seed)
        ids = list(range(0, n_datos))
        random.shuffle(ids)
        particiones = []

        for i in range(0, self.k):
            particion = Particion()
            print(i*datos_por_fold)
            particion.indicesTest = ids[i*datos_por_fold: (i+1)*datos_por_fold]
            if i != 0:
                particion.indicesTrain += ids[(i-1)*datos_por_fold: i*datos_por_fold]

            if i != self.k-1:
                particion.indicesTrain += ids[(i+1)*datos_por_fold:]

            particiones.append(particion)

        return particiones