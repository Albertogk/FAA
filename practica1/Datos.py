import pandas as pd
import numpy as np

class Datos:

    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):
        self.nominalAtributos = []
        self.datos = pd.read_csv(nombreFichero)
        self.datos = self.datos.to_numpy()
        self.diccionario = []

        aux_set = []

        for atributo in self.datos[0]:
            self.nominalAtributos.append(self.esNominal(atributo))
            aux_set.append(set())

        for linea in self.datos:
            idx = 0
            for atributo in linea:
                if self.nominalAtributos[idx]:
                    aux_set[idx].add(atributo)

                idx += 1

        for i in range(len(self.datos[0])):
            self.diccionario.append({item: val for val, item in enumerate(sorted(aux_set[i]))})

    def esNominal(self, atributo):
        try:
            float(atributo)
            return False
        except ValueError:
            return True

    # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        retorno = []
        for i in idx:
            retorno.append(self.datos[i])

        return retorno


