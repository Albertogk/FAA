import numpy as np

lista = np.array([[7, 8, 9]])
matriz = np.array([[1, 2, 3], [4, 5, 6]])

matriz = np.append(matriz, lista, axis=0)

print(matriz)
