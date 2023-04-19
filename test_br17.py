import numpy as np
import ACO


data = np.loadtxt('br17.txt', dtype = float)

# istanzio
br17 = ACO.TCProblem(data, 'EXPLICIT')

# colonia
br17.antColony(2, 2.5, 0.8, 100)

# stop condition
br17.set_stop_condition('ITER', 1000)

br17.solve()

print(br17.results.shortestPath)
print(br17.results.shortestTour)


