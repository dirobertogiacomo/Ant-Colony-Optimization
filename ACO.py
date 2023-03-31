"""
Description of the module

Giacomo Di Roberto, March 2023, version 1.0
"""
import numpy as np
import random as rd

class AntColony:
    """
    Class that define an ant colony

    ...

    Attributes
    ----------
    alpha : int
        Parameter to control the weight of the pheromone
    beta : int
        Parameter to control the weight of the heuristic visibility
    p : int
        Parameter to control the pheromone decay
    m : int
        Number of ants
    N : int
        Number of cities
    iter_max : int
        maximum number of iterations
    Q : int, optional
        arbitrary constant to control tour lenght (default is 1)


    Methods
    -------
    initialize(self)
        Initializes the algorithm


        
    Giacomo Di Roberto, March 2023, version 1.0

    """
    # Attributes
    def __init__(self, alpha: int, beta: int, p: int, m: int,N: int, iter_max: int, Q=1) -> object:
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.m = m
        self.N = N
        self.iter_max = iter_max
        self.Q = Q

        if m < N:
            raise AttributeError('The number of ants, m, must be greater or equal than the number of cities, N')
    
    # Methods
    def initialize(self, city_matrix: np.ndarray):
        """
        Initializes the algorithm


        """
        
        def compute_lenght(N: int, M: np.ndarray):
            """
            Computes the Euclidian distance between the cities

            ...

            Parameters
            ---------
            N : int
                Number of cities
            M : ndarray
                Cities coordinate matrix

            Return
            ------
            distance : nparray
                Cities distance matrix 

            """

            distance = np.zeros((N,N), int) # preallocating the matrix

            # computing the Euclidian distance
            # xd = x[i] - x[j];
            # yd = y[i] - y[j];
            # dij = nint( sqrt( xd*xd + yd*yd) );
            for i in range(N):
                m = np.power(np.subtract(M[i,:], M), 2)
                m = np.round(np.sqrt(np.sum(m,axis=1)))
                distance[i, :] = m
            
            return distance

        def compute_pheromone(N: int):
            """
            Generates the initial matrix of the pheromone distribution

            Parameters
            ----------
            N : int
                Number of cities

            Return
            ------
            pheromone : ndarray
                Initial matrix of the pheromone distribution 
            
            """
            
            di = np.diag_indices(N)
            pheromone = np.ones((N,N), int)
            pheromone[di] = 0

            return pheromone

        def ants_position(m: int, N: int):
            """
            Returns an array containing the number of ants for each city

            ...

            Parameters
            ---------
            m : int
                Number of ants
            N : int
                Number of cities
            
            Return
            -----
            ants_on_city : ndarray
                array containing the number of ants for each city

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


        # number of rows in the input city matrix must be equal to the number of cities N assigned to the object
        if city_matrix.shape[0] != self.N:
            raise AttributeError('The number of cities, N, doesn\'t match the matrix shape. N and the number of rows in the matrix must be the same') 
        
        # computing the initial pheromone distribution matrix
        pheromone = compute_pheromone(self.N)

        # computing the matrix distance 
        distance = compute_lenght(self.N, city_matrix)

        # assigning the ants in the cities
        ants_on_city = ants_position(self.m, self.N)  


        return pheromone, distance, ants_on_city


