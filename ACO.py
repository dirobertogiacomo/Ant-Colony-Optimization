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
    alpha : float
        Parameter to control the weight of the pheromone
    beta : float
        Parameter to control the weight of the heuristic visibility
    p : float
        Parameter to control the pheromone decay
    m : int
        Number of ants
    N : int
        Number of cities
    iter_max : int
        Maximum number of iterations
    Q : int, optional
        Arbitrary constant to control tour lenght (default is 1)


    Methods
    -------
    initialize(self)
        Initializes the algorithm


        
    Giacomo Di Roberto, March 2023, version 1.0

    """
    # Attributes
    def __init__(self, alpha: float, beta: float, p: float, m: int, N: int, iter_max: int, Q=1) -> object:
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

        ...

        Parameters
        ----------
        city_matrix : ndarray
            Cities coordinate matrix
        
        Return
        ------
            pheromone : ndarray
                Initial matrix of the pheromone distribution
            distance : ndarray
                Cities distance matrix
            ants_on_city : ndarray
                Array containing the number of ants for each city

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
            pheromone = np.ones((N,N))
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
                Array containing the number of ants for each city

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

        ######

        # number of rows in the input city matrix must be equal to the number of cities N assigned to the object
        if city_matrix.shape[0] != self.N:
            raise AttributeError('The number of cities, N, doesn\'t match the matrix shape. N and the number of rows in the matrix must be the same') 
        
        # computing the initial pheromone distribution matrix
        pheromone = compute_pheromone(self.N)

        # computing the matrix distance 
        distance = compute_lenght(self.N, city_matrix)

        # assigning the ants in the cities
        ants_on_city = ants_position(self.m, self.N)  


        return (pheromone, distance, ants_on_city)
    
    def loop(self, init):

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
        
        def compute_path_length(distance, path, start):
            
            length = 0
            j = start
            for i in np.nditer(path):
                length += distance[j, i]
                j = i
            
            return length

        ######

        # initialising the variables
        pheromone_distr, distance, ants_on_city = init
        #A = pheromone_distr
        visibility = 1/distance 
        # distance vector
        path_lengths = np.zeros(self.m)
        
        shortest_tour = 99999999999999999

        iter = 0

        delta_t = np.zeros((self.N, self.N))

        while(iter < self.iter_max):

            # for each city
            for city in range(self.N):

                # number of ants in the city
                N_ants = ants_on_city[city]

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
                        edge_pheromone = pheromone_distr.copy()[current_city, :]
                        edge_pheromone[not_allowed_cities] = 0

                        # compute probability
                        probability = compute_probability(edge_visibility, edge_pheromone, self.alpha, self.beta)

                        # choosing the best city
                        next_city = np.argmax(probability)

                        # updating the path distance
                        path[edge] = distance.copy()[current_city, next_city]
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
                    #path_lengths[city] = np.sum(path) + distance[current_city, city]
                    path_lengths[city] = compute_path_length(distance, percorso, city)

                    # updating delta_t
                    delta_t += delta_t_k/path_lengths[ant]
                
                # updating the shortest tour
                tour = np.min(path_lengths[np.nonzero(path_lengths)])
                if tour < shortest_tour:
                    shortest_tour = tour
            
            # computing new pheromone distribution matrix
            pheromone_distr = ((1-self.p)*pheromone_distr) + delta_t

            iter += 1

        return shortest_tour, pheromone_distr


