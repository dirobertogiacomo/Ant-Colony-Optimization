import numpy as np
import matplotlib.pyplot as plt
import ACO

data = np.loadtxt('berlin52.txt', usecols = (1,2), dtype = int)

optimalTour = 7542

# istanzio il problema
berlin52 = ACO.TCProblem(data, 'EUC_2D')

# genero la colonia
berlin52.antColony(3, 3, 0.2, 30)

# setto la condizione di stop
berlin52.set_stop_condition('ITER', 100)

# risolvo il problema
berlin52.solve()

shortestPath = berlin52.results.shortestPath 

print('Shortest tour: ', berlin52.results.shortestTour)
print('Error[%]: ', (berlin52.results.shortestTour - optimalTour)/optimalTour*100)


# plot
for i in range(len(shortestPath)-1):
    currentCity = shortestPath[i]
    nextCity = shortestPath[i+1]
    x = [data[currentCity, 0], data[nextCity , 0]]
    y = [data[currentCity, 1], data[nextCity, 1]]
    plt.plot(x,y, color = '#bcbcbc')

# last city
x = [data[shortestPath[-1], 0], data[shortestPath[0] , 0]]
y = [data[shortestPath[-1], 1], data[shortestPath[0], 1]]
plt.plot(x,y,color = '#bcbcbc')

plt.scatter(data[:,0], data[:, 1])

plt.show()
