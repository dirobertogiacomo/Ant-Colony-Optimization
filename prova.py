import numpy as np
import random as rd

def compute_probability(visibility: np.ndarray, pheromone: np.ndarray, alpha: float, beta: float):
            """
            Compute the probability array for one edge

            ...

            Parameters
            ----------
            visibility : 1-D ndarray
                1-D array of the heuristic visibility of the edge
            pheromone : 1-D ndarray
                1-D array of the pheromone distribution as seen from the edge
            alpha : float
                Alpha coefficient
            beta : float
                Beta coefficient

            Return
            ------
            probability : 1-D array
                Probability array for the edge

            """

            # checking parameters
            if (visibility.ndim != 1) or (pheromone.ndim != 1):
                raise TypeError('Expected 1-D array, but multidimensional array was found')
            if visibility.size != pheromone.size:
                raise ValueError('Input arrays must have the same length')

            # length of the array
            L = visibility.size
            # preallocating probability array
            probability = np.zeros(L)

            # computing denominator
            D = np.sum(np.prod([np.power(pheromone, alpha), np.power(visibility, beta)], axis=0))

            # computing probability array
            for i in range(L):
                # computing numerator
                N = (pheromone[i]**alpha)*(visibility[i]**beta)
                probability[i] = N/D

            return probability

distance = np.array([[999,3,6,2,3],[3,999,5,2,3],[6,5,999,6,4],[2,2,6,999,6],[3,3,4,6,999]])
visibility = 1/distance
pheromone = np.ones((5,5))
pheromone[np.diag_indices(5)] = 0
N = 5
initial_city = 3
# creating the initial filter arrays
allowed_cities = [True]*N
allowed_cities[initial_city] = False

not_allowed_cities = [False]*N
not_allowed_cities[initial_city] = True

path = [int]*(N-1)
city = initial_city

for edge in range(N - 1):

    
    edge_visibility = visibility[city, :]
    #edge_visibility[not_allowed_cities] = 0
    np.putmask(edge_visibility, allowed_cities, 0)
    edge_pheromone = pheromone[city, :]
    #edge_pheromone[not_allowed_cities] = 0
    np.putmask(edge_pheromone, allowed_cities, 0)

    # compute probability
    #probability = compute_probability(edge_visibility[allowed_cities],edge_pheromone[allowed_cities], 0.7, 0.7)
    probability = compute_probability(edge_visibility,edge_pheromone, 0.7, 0.7)
    

    # scelgo probabilità maggiore
    idx = np.argmax(probability)

    city = idx
    path[edge] = city

    allowed_cities[idx] = False
    not_allowed_cities[idx] = True

#print(probability)
#print(np.sum(probability))
print(path)

print(probability)
print(probability/2)

probability += probability/2
print(probability)