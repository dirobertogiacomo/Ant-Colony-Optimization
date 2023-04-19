import numpy as np
import ACO

data = np.loadtxt('bier127.txt', usecols = (1,2), dtype = float)

optimalTour = 118282

# istanzio il problema
bier127 = ACO.TCProblem(data, 'EUC_2D')

# genero la colonia
bier127.antColony(3, 3, 0.2, 20)

# setto la condizione di stop
bier127.set_stop_condition('ITER', 100)

# risolvo il problema
bier127.solve()

print(bier127.results.shortestTour)
print(bier127.results.computationTime)
print('Error[%]: ', (bier127.results.shortestTour - optimalTour)/optimalTour*100)

