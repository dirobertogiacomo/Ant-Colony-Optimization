import ACO
import numpy as np

distance = np.array([[999,3,6,2,3],[3,999,5,2,3],[6,5,999,6,4],[2,2,6,999,6],[3,3,4,6,999]])

alpha = 0.7
beta = 0.7
p = 0.6
m = 5
iter_max = 1

# definisco la colonia
antcolony = ACO.AntColony(alpha, beta, p, m)
# istanzio il problema
problem = ACO.TCProblem('EXPLICIT', distance)
# setto la condizione di stop
problem.set_stop_condition('ITER', iter_max)

# risolvo il problema
shortest_tour, shortest_path = problem.solve(antcolony)

print(shortest_tour, shortest_path)

