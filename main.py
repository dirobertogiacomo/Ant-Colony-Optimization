import ACO
import numpy as np
import matplotlib.pyplot as plt

alpha = 1
beta = 1
p = 1
m = 10
N = 3
iter_max = 10
pippo = ACO.AntColony(alpha, beta, p, m, N, iter_max)

matrix = np.array([[1,2], [4,7], [7,3]])
a,b,c = pippo.initialize(matrix)
print(a,b,c)

plt.scatter(matrix[:,0], matrix[:,1])
plt.show()
