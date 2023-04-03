"""
Description of the module

Giacomo Di Roberto, March 2023, version 1.1
"""
import numpy as np
import random as rd

# keywords
STOP_CONDITION = 'ITER'
#STOP_CONDITION = ['ITER', 'TIME', 'CONVERGENCE']
TYPE_MATRIX = 'EXPLICIT'
#TYPE_MATRIX = ['EXPLICIT', 'EUC_2D']

class AntColony:
    """
    Class that define an ant colony

    ...

    Attributes
    ----------
    alpha : float
        Parameter to control the weight of the pheromone
    beta : float
        Parameter to control the weight of the heuristic visibility
    p : float
        Parameter to control the pheromone decay
    m : int
        Number of ants
    Q : int, optional
        Arbitrary constant to control tour lenght (default is 1)

    """

    def __init__(self, alpha: float, beta: float, p: float, m: int, Q=1) -> object:
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.m = m
        self.Q = Q


class TCProblem:
    """
    Class that define a simmetric Travel Salesman Problem

    Attributes
    ----------
    N : int
        Number of cities
    distance : np.ndarray
        Cities distance matrix
    stop_condition

    condition

    pheromone:

    ants_on_city:

    shortest_path

    shortest_tour
    
    Methods
    -------
    set_stop_condition

    solve

    """

    def __init__(self, type: str, M: np.ndarray) -> object:
        if type == TYPE_MATRIX:
            self.distance = M
            self.N = M.shape[0]
        else:
            raise NameError
        
        # initialising the other attributes
        self.stop_condition = None
        self.condition = None
        self.pheromone = None
        self.ants_on_city = None
        self.shortest_path = None
        self.shortest_tour = None
            
    
    def set_stop_condition(self, condition: str, n: int):
        """
        Set the stop condition

        """

        if STOP_CONDITION == condition:
            self.stop_condition = condition
            self.condition = n
        else:
            raise NameError
    
    def solve(self, Ants: AntColony):
        """
        Solve the problem

        """

        def initialize():
            """
            Initializes the algorithm

            """
        
            def compute_pheromone(N: int):
                """
                Generates the initial matrix of the pheromone distribution
                
                """
                
                di = np.diag_indices(N)
                pheromone = np.ones((N,N))
                pheromone[di] = 0

                return pheromone

            def ants_position(m: int, N: int):
                """
                Returns an array containing the number of ants for each city

                """

                # preallocating the matrix, including one ant for each city
                ants_on_city = np.ones(N, int)
                if m != N:
                    # randomly assigning the remaining ants
                    remaining_ants = m-N
                    for i in range(N):
                        a = rd.randint(0,remaining_ants)
                        ants_on_city[i] += a
                        remaining_ants -= a

                return ants_on_city
            
            # computing the initial pheromone distribution matrix
            pheromone = compute_pheromone(self.N)

            # assigning the ants in the cities
            ants_on_city = ants_position(Ants.m, self.N)

            self.pheromone = pheromone
            self.ants_on_city = ants_on_city
        
        def loop():

            def compute_probability(visibility: np.ndarray, pheromone: np.ndarray, alpha: float, beta: float):
                """
                Compute the probability array for one edge

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
            
            def compute_path_length(distance, path, start):
                
                length = 0
                j = start
                for i in np.nditer(path):
                    length += distance[j, i]
                    j = i
                
                return length
            
            def check_stop_condition(N:int):
                if N < self.condition:
                    condizione = True
                else:
                    condizione = False
                return condizione

            ######

            # initialising the variables
            visibility = 1/self.distance 

            path_lengths = np.zeros(Ants.m)
            
            shortest_tour = 99999999999999999
            shortest_path = np.zeros(self.N, int)

            iter = 0

            delta_t = np.zeros((self.N, self.N))

            condizione = True

            while(condizione):

                # for each city
                for city in range(self.N):

                    # number of ants in the city
                    N_ants = self.ants_on_city[city]

                    # for each ant in the city
                    for ant in range(N_ants):

                        # creating the initial filter array
                        not_allowed_cities = [False]*self.N
                        not_allowed_cities[city] = True

                        # preallocating the path vector
                        path = np.zeros(self.N - 1)
                        percorso = np.zeros(self.N, int)
                        percorso[self.N - 1] = city
                        

                        # preallocating delta_t_k
                        delta_t_k = np.zeros((self.N, self.N))
                        
                        # initialising the path
                        current_city = city

                        # for each edge
                        for edge in range(self.N - 1):
        
                            # computing visibility pheromone distribution as seen from the edge
                            edge_visibility = visibility.copy()[current_city, :]
                            edge_visibility[not_allowed_cities] = 0
                            edge_pheromone = self.pheromone.copy()[current_city, :]
                            edge_pheromone[not_allowed_cities] = 0

                            # compute probability
                            probability = compute_probability(edge_visibility, edge_pheromone, Ants.alpha, Ants.beta)

                            # choosing the best city
                            next_city = np.argmax(probability)

                            # updating the path 
                            path[edge] = self.distance.copy()[current_city, next_city]
                            percorso[edge] = next_city

                            # updating the filters
                            not_allowed_cities[next_city] = True

                            current_city = next_city
                        
                        # computing delta_t_k
                        j = city
                        for i in np.nditer(percorso):
                            delta_t_k[j, i] = 1
                            delta_t_k[i, j] = 1
                            j=i

                        # computing total path length for the ant
                        path_lengths[city] = compute_path_length(self.distance, percorso, city)

                        # updating delta_t
                        delta_t += delta_t_k/path_lengths[ant]
                    
                    # updating the shortest tour
                    tour = np.min(path_lengths[np.nonzero(path_lengths)])
                    if tour < shortest_tour:
                        shortest_tour = tour
                        shortest_path = percorso
                
                # computing new pheromone distribution matrix
                self.pheromone = ((1-Ants.p)*self.pheromone) + delta_t

                # checking stop condition
                iter += 1
                condizione = check_stop_condition(iter)
            
            self.shortest_path = shortest_path
            self.shortest_tour = shortest_tour 
        
        #############################################

        # initializing the algorithm
        initialize()

        # loop
        loop()
        
        return self.shortest_tour, self.shortest_path
    