import ACO
import numpy as np
import matplotlib.pyplot as plt

distance = np.array([[999,3,6,2,3],[3,999,5,2,3],[6,5,999,6,4],[2,2,6,999,6],[3,3,4,6,999]])

alpha = 0.7
beta = 0.7
p = 0.6
m = 5
iter_max = 50
time = 0.5


# istanzio il problema
problem = ACO.TCProblem(distance, 'EXPLICIT')

# definisco la colonia
problem.antColony(alpha, beta, p, m)

# setto la condizione di stop
problem.set_stop_condition('TIME', time)

# risolvo il problema
problem.solve()

print(problem.results.shortestPath)
print(problem.results.shortestTour)
print(problem.results.iter)
