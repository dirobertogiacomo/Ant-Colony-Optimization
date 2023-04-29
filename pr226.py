import numpy as np
import matplotlib.pyplot as plt
import ACO

problem_name = 'pr226'

data = np.loadtxt(problem_name+'.txt', usecols = (1,2), dtype = int)

optimalTour = 80369

N = 20

path = [None]*N
tour = [None]*N
error = [None]*N

# set the problem
pr226 = ACO.TCProblem(data, 'EUC_2D')

# ant colony
pr226.antColony(2, 2, 0.1, 10)

# set stop condition
pr226.set_stop_condition('ITER', 150)

# solve the problem
for i in range(N):
    print(i)
    pr226.solve()
    path[i] = pr226.results.shortestPath
    tour[i] = pr226.results.shortestTour
    error[i] = (pr226.results.shortestTour - optimalTour)/optimalTour

shortestTour = min(tour)
averageTour = sum(tour)/len(tour)
shortestPath = path[tour.index(shortestTour)]

# mean error
mean_error = (1/N)*sum(error)*100


# save results
txt = '{},{},{},{},{},{}\n'.format(problem_name,pr226.N,optimalTour,shortestTour,averageTour,mean_error)
f = open('results.csv','a')
f.write(txt)
f.close()


# plot
for i in range(len(shortestPath)-1):
    currentCity = shortestPath[i]
    nextCity = shortestPath[i+1]
    x = [data[currentCity, 0], data[nextCity , 0]]
    y = [data[currentCity, 1], data[nextCity, 1]]
    plt.plot(x,y, color = 'k', linewidth = '0.5')

# last city
x = [data[shortestPath[-1], 0], data[shortestPath[0] , 0]]
y = [data[shortestPath[-1], 1], data[shortestPath[0], 1]]
plt.plot(x,y,color = 'k', linewidth = '0.5')

plt.scatter(data[:,0], data[:, 1], color = 'k', s = 25)

plt.suptitle(problem_name)
title_string = 'Length: {}'.format(shortestTour)
plt.title(title_string)

plt.show()

