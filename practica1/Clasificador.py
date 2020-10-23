from abc import ABCMeta, abstractmethod
import numpy as np

class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self, datos, pred):
        false_c1 = 0
        false_c2 = 0

        i = 0
        for fila in datos:

            if fila[-1] != pred[i]:
                if pred[i] == self.atributos[-1][0]:
                    false_c1 += 1

                #tener en cuenta que error
                else:
                    false_c2 += 1
            i += 1

        return (false_c1 + false_c2)/len(datos)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opcion es repetir la validación simple un número especificado de veces, obteniendo en cada una un error. Finalmente se calcularia la media.
        pass

    ##############################################################################


class ClasificadorNaiveBayes(Clasificador):

    def __init__(self):
        self.datos_atributo = []
        self.atributos = []
        self.n_valores = []
        self.n_c1 = 0
        self.n_c2 = 0

    def obtener_atributos(self, datostrain):
        aux_set = []
        n_valores = []

        for _ in datostrain[0]:
            aux_set.append(set())

        for linea in datostrain:

            idx = 0
            for atributo in linea:
                aux_set[idx].add(atributo)
                idx += 1

        for i in range(len(aux_set)):
            n_valores.append(len(aux_set[i]))

        atributos = []

        for i in range(len(aux_set)):
            atributos.append(sorted(aux_set[i]))

        self.atributos = atributos
        self.n_valores = n_valores

        return n_valores, atributos

    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        n_valores, atributos = self.obtener_atributos(datostrain)

        probabilidades = []

        for i in range(len(atributos)):
            probabilidades.append([])

        n_c1 = 0
        n_c2 = 0

        for fila in datostrain:
            if fila[-1] == atributos[-1][0]:
                n_c1 += 1
            else:
                n_c2 += 1

        idx_atributo = 0


        for atributo in atributos:

            for valor in atributo:

                cont_c1 = 0
                cont_c2 = 0
                for fila in datostrain:
                    if fila[idx_atributo] == valor:
                        if fila[-1] == atributos[-1][0]:
                            cont_c1 += 1
                        else:
                            cont_c2 += 1

                a = cont_c1/n_c1
                b = cont_c2/n_c2

                probabilidades[idx_atributo].append((a, b))

            idx_atributo += 1

        self.datos_atributo = probabilidades
        self.n_c1 = n_c1
        self.n_c2 = n_c2

        return probabilidades, (n_c1, n_c2)

    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):

        result = []

        for fila in datostest:
            p1 = 1
            p2 = 1
            flag = 0
            cont = 0
            for i in range(len(fila)-1):

               '''try:
                    idx = self.atributos[i].index(fila[i])
                except ValueError:
                    flag = 1
                    break'''

               idx = fila[i]
               p1 *= self.datos_atributo[i][idx][0]
               p2 *= self.datos_atributo[i][idx][1]

               cont += 1

            p1 *= self.n_c1/(self.n_c1 + self.n_c2)
            p2 *= self.n_c2/(self.n_c1 + self.n_c2)


            p1_aux = p1/(p1+p2)
            p2 = p2/(p1+p2)
            
            p1 = p1_aux

            if flag == 1:
                result.append('error')
            else:

                print([p1, p2])
                if p1 >= p2:
                    result.append(self.atributos[-1][0])
                else:
                    result.append(self.atributos[-1][1])


        return result