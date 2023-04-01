import numpy as np

matrice = np.arange(25).reshape(5,5)
matrice_copia = matrice.copy()
filtro = [False]*5
filtro[2] = True

print('Matrice prima: ')
print(matrice)
print(matrice_copia)

for i in range(5):
    riga = matrice_copia[i, :]
    riga[filtro] = 0

print('Matrrice dopo: ')
print(matrice)
print(matrice_copia)
