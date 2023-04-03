import ACO
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.7
beta = 0.7
p = 0.6
m = 5
N = 5
iter_max = 1
pippo = ACO.AntColony(alpha, beta, p, m, N, iter_max)
distance = np.ones((N,N))
pheromone, distance, ants_on_city = pippo.initialize(distance)

distance = np.array([[999,3,6,2,3],[3,999,5,2,3],[6,5,999,6,4],[2,2,6,999,6],[3,3,4,6,999]])

init = (pheromone, distance, ants_on_city)

for i in range(len(init)):
    print(init[i])

resoult = pippo.loop(init)

for i in range(len(resoult)):
    print(resoult[i])
