"""
Description of the module

Giacomo Di Roberto, March 2023, version 1.2
"""
import numpy as np
import random as rd
import time

# constats
STOP_CONDITION = ['ITER', 'TIME']
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
        Identifies the type of stop condition. Could be 'ITER' or 'TIME'.

    stopConditionValue : int
        Identifies the value given to the stop condition (number of seconds or number of iterations)

    pheromone : np.ndarray
        Pheromene distribution matrix 

    ants_on_city : np.ndarray
        Array containing the number of ants for each city

    results : class
        Subclass to store the results and plot the graphs

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
    
    Methods
    -------
    set_stop_condition(condition_type, condition_value)
        Set the stop condition

    antColony(alpha, beta, p, m, Q)
        Define an ant colony

    solve()
        Solve the problem

    """

    # subcalss

    class results():
        """
        Subclass to store the results

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
            
        """

        def __init__(self) -> None:
            self.shortestPath = None
            self.shortestTour = None
            self.iter = None
            self.computationTime = None

    # ##################
    # TCProblems methods
    # ##################

    def __init__(self, M: np.ndarray, type: str):

        # #############################
        # nested function for __init__
        # #############################

        def compute_distance(N: int, Matrix: np.ndarray):
            """
            Computes the Euclidian distance between the cities

            """

            # preallocating the matrix
            distance = np.zeros((N,N), int) 

            # computing the Euclidian distance
            # xd = x[i] - x[j];
            # yd = y[i] - y[j];
            # dij = nint( sqrt( xd*xd + yd*yd) );
            for i in range(N):
                m = np.power(np.subtract(Matrix[i,:], Matrix), 2)
                m = np.round(np.sqrt(np.sum(m,axis=1)))
                distance[i, :] = m
            
            # avoiding divisions by zero
            di = np.diag_indices(N)
            distance[di] = 1000000
            
            return distance
        

        # #################
        # body of __init__
        # #################


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
        self.results = self.results()
        self.alpha = None
        self.beta = None
        self.p = None
        self.m = None
        self.Q = 1
    
            
    def set_stop_condition(self, condition_type: str, condition_value: int) -> None:
        """
        Set the stop condition

        Parameters
        ----------
        condition_type : string
            Type of the stop condition. Could be 'ITER' or 'TIME'. 

        condition_value : int
            value given to the stop condition (number of seconds or number of iterations)

        """

        if condition_type in STOP_CONDITION :
            self.stopCondition = condition_type
            self.stopConditionValue = condition_value
        else:
            raise NameError
    
    def antColony(self, alpha: float, beta: float, p: float, m: int, Q=1) -> None:
        """
        Define an ant colony

        Parameters
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

        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.m = m
        self.Q = Q

    
    def solve(self) -> None:
        """
        Solve the problem
            
        """

        # ############################
        # nested functions for solve()
        # ############################

        def initialize():
            """
            Initializes the algorithm

            """

            # #################################
            # nested functions for initialize()
            # #################################

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
                m = self.m
                
                # randomly assign the ants on the cities
                ants_on_city = np.zeros(N, int)

                remaining_ants = m
                while remaining_ants > 0:
                    # randomly choise a city
                    city = rd.choice(range(N))
                    # assign to this city one ant
                    ants_on_city[city] += 1
                    remaining_ants -= 1

                return ants_on_city
            
            # #####################
            # body of initialize()
            # #####################

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

            # ###########################
            # nested functions for loop()
            # ###########################

            def compute_probability(visibility: np.ndarray, pheromone: np.ndarray):
                """
                Compute the probability array for one edge

                """

                alpha = self.alpha
                beta = self.beta

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
            
            def check_stop_condition():
                """
                Check if the stop condition of the problem is True

                """

                if self.stopCondition == 'ITER':
                    if self.results.iter < self.stopConditionValue:
                        return True
                    else:
                        return False
                    
                if self.stopCondition  == 'TIME' :
                    if self.results.computationTime < self.stopConditionValue:
                        return True
                    else:
                        return False

            # ##############        
            # body of loop()
            # ##############

            
            # initialising the variables
            visibility = 1/self.distance # computational visibility

            cities_containing_ants = np.nonzero(self.ants_on_city)[0]  # cities containing ants 

            #path_lengths = np.zeros(self.m)
            path_lengths = np.zeros(np.size(cities_containing_ants))
            
            shortestTour = 99999999999999999
            shortestPath = np.zeros(self.N, int)

            self.results.iter = 0

            total_pheromoneDrop = np.zeros((self.N, self.N))

            isTrue = True

            start = time.time()

            while(isTrue):

                idx = 0
                # for each city containing ants
                for city in cities_containing_ants: 

                    # number of ants in the city
                    N_ants = self.ants_on_city[city]

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
                        ant_pheromoneDrop[j, i] = self.Q
                        ant_pheromoneDrop[i, j] = self.Q
                        j=i

                    # computing total path length for the ant
                    path_lengths[idx] = compute_path_length(path, start=city)

                    # updating total_pheromoneDrop
                    total_pheromoneDrop += (ant_pheromoneDrop/path_lengths[idx])*N_ants
                    
                    # updating the shortest tour
                    tour = np.min(path_lengths[np.nonzero(path_lengths)])
                    if tour < shortestTour:
                        shortestTour = tour
                        shortestPath = path
                        print(shortestTour)

                    idx += 1
                    
                # computing new pheromone distribution matrix
                self.pheromone = ((1-self.p)*self.pheromone) + total_pheromoneDrop

                # updating the flags
                self.results.iter += 1
                self.results.computationTime = time.time() - start

                # checking stop condition
                isTrue = check_stop_condition()
            
            # saving the results
            self.results.shortestPath = shortestPath
            self.results.shortestTour = shortestTour
        

        # ################
        # body of solve()
        # ################


        print('Starting the algorithm...\n')

        # initializing the algorithm
        initialize()

        print('Loop...\n')

        # loop
        loop()

        print('100% Complete\n')

    