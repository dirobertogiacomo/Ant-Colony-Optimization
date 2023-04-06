"""
Description of the module

Giacomo Di Roberto, March 2023, version 1.1
"""
import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

# constats
STOP_CONDITION = 'ITER'
#STOP_CONDITION = ['ITER', 'TIME', 'CONVERGENCE']
TYPE_MATRIX = ['EXPLICIT', 'EUC_2D']

class TCProblem:
    """
    Class that define a simmetric Travel Salesman Problem

    ...

    Parameters
    ----------
    M : np.ndarray
        Edge weigths (or distances) matrix. Must be a square matrix
    
    type : str
        Specifies how the edge weights (or distances) are given (according to TSPLIB 95)
            'EXPLICIT': Weights are listed explicitly in the corresponding section
            'EUC 2D': Weights are Euclidean distances in 2-D

    Attributes
    ----------
    N : int
        Number of cities

    distance : np.ndarray
        Edge weigths (or distances) matrix

    stopCondition : string
        Identifies the type of stop condition 

    stopConditionValue : int

    pheromone:

    ants_on_city:
    
    Methods
    -------
    set_stop_condition(condition_type, condition_value)

    solve(ant_colony)

    """

    # subcalsses
    
    class antColony:
        """
        Subclass that define an ant colony

        ...

        Parameters & Attributes
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

        def __init__(self) -> None:
            pass
        
        def __call__(self, alpha: float, beta: float, p: float, m: int, Q=1) -> None:
            self.alpha = alpha
            self.beta = beta
            self.p = p
            self.m = m
            self.Q = Q
        

    class results:
        """
        Subclass to store the results and plot the graphs

        ...

        Attributes
        ----------
        shortestPath : list
            Path of the shortest tour

        shortestTour : int
            Length of the shortest tour
        
        iter : int
            Number of iteration
        
        computationTime : float
            Computation time of the algorithm
        
        Methods
        -------
        plot()

        """

        def __init__(self) -> None:
            self.shortestPath = None
            self.shortestTour = None
            self.iter = None
            self.computationTime = None
    
    # 

    def __init__(self, M: np.ndarray, type: str):

        def compute_distance(N: int, Matrix: np.ndarray):
            """
            Computes the Euclidian distance between the cities

            """

            distance = np.zeros((N,N), int) # preallocating the matrix

            # computing the Euclidian distance
            # xd = x[i] - x[j];
            # yd = y[i] - y[j];
            # dij = nint( sqrt( xd*xd + yd*yd) );
            for i in range(N):
                m = np.power(np.subtract(Matrix[i,:], Matrix), 2)
                m = np.round(np.sqrt(np.sum(m,axis=1)))
                distance[i, :] = m
            
            return distance
        
        if type in TYPE_MATRIX:
            self.N = M.shape[0]
            if type == 'EUC_2D':
                self.distance = compute_distance(self.N, M)
            else:
                self.distance = M
        else:
            raise NameError
        
        # initialising the other attributes
        self.stopCondition = None
        self.stopConditionValue = None
        self.pheromone = None
        self.ants_on_city = None
        self.antColony = self.antColony()
        self.results = self.results()
            
    def set_stop_condition(self, condition_type: str, condition_value: int):
        """
        Set the stop condition

        """

        if STOP_CONDITION == condition_type:
            self.stopCondition = condition_type
            self.stopConditionValue = condition_value
        else:
            raise NameError
    
    def solve(self):
        """
        Solve the problem

        ...

        Parameters
        ----------
        Ants : AntColony object 
            
        """

        # 

        def initialize():
            """
            Initializes the algorithm

            """

            # 

            def compute_pheromone():
                """
                Generates the initial matrix of the pheromone distribution
                
                """
                
                di = np.diag_indices(self.N)
                pheromone = np.ones((self.N, self.N))
                pheromone[di] = 0

                return pheromone

            def ants_position():
                """
                Returns an array containing the number of ants for each city

                """

                N = self.N
                m = self.antColony.m

                # preallocating the array, including one ant for each city
                ants_on_city = np.ones(N, int)
                if m != N:
                    # randomly assigning the remaining ants
                    remaining_ants = m-N
                    while remaining_ants > 0:
                        for i in range(N):
                            a = rd.randint(0, remaining_ants)
                            ants_on_city[i] += a
                            remaining_ants -= a

                return ants_on_city
            
            # body of initialize()
            
            # computing the initial pheromone distribution matrix
            pheromone = compute_pheromone()

            # assigning the ants in the cities
            ants_on_city = ants_position()

            # saving the results
            self.pheromone = pheromone
            self.ants_on_city = ants_on_city

        def loop():
            """
            Loop of the algorithm

            """
            # 

            def compute_probability(visibility: np.ndarray, pheromone: np.ndarray):
                """
                Compute the probability array for one edge

                """

                alpha = self.antColony.alpha
                beta = self.antColony.beta

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
            
            def compute_path_length(path, start):
                """
                Compute the length of the given path

                """
                
                length = 0
                j = start
                for i in np.nditer(path):
                    length += self.distance[j, i]
                    j = i
                
                return length
            
            def check_stop_condition(N: int):
                """
                Check if the stop condition of the problem is True

                """

                if N < self.stopConditionValue:
                    return True
                else:
                    return False 

            # body loop()

            start = time.time()

            # initialising the variables
            visibility = 1/self.distance 

            path_lengths = np.zeros(self.antColony.m)
            
            shortestTour = 99999999999999999
            shortestPath = np.zeros(self.N, int)

            self.results.iter = 0

            total_pheromoneDrop = np.zeros((self.N, self.N))

            isTrue = True

            while(isTrue):

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
                        path = np.zeros(self.N, int)
                        path[self.N - 1] = city
                        

                        # preallocating ant_pheromoneDrop
                        ant_pheromoneDrop = np.zeros((self.N, self.N))
                        
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
                            probability = compute_probability(edge_visibility, edge_pheromone)

                            # choosing the best city
                            next_city = np.argmax(probability)

                            # updating the path 
                            path[edge] = next_city

                            # updating the filters
                            not_allowed_cities[next_city] = True

                            current_city = next_city
                        
                        # computing ant_pheromoneDrop
                        j = city
                        for i in np.nditer(path):
                            ant_pheromoneDrop[j, i] = 1
                            ant_pheromoneDrop[i, j] = 1
                            j=i

                        # computing total path length for the ant
                        path_lengths[city] = compute_path_length(path, city)

                        # updating total_pheromoneDrop
                        total_pheromoneDrop += ant_pheromoneDrop/path_lengths[ant]
                    
                    # updating the shortest tour
                    tour = np.min(path_lengths[np.nonzero(path_lengths)])
                    if tour < shortestTour:
                        shortestTour = tour
                        shortestPath = path
                
                # computing new pheromone distribution matrix
                self.pheromone = ((1-self.antColony.p)*self.pheromone) + total_pheromoneDrop

                # updating the flags
                self.results.iter += 1
                global stop
                stop = time.time() - start

                # checking stop condition
                isTrue = check_stop_condition(self.results.iter)
            
            # saving the results
            self.results.shortestPath = shortestPath
            self.results.shortestTour = shortestTour
        
        # body of solve()

        # initializing the algorithm
        initialize()

        # loop
        loop()


    