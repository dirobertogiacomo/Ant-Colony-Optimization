import numpy as np
import random as rd

m = 5
N = 3

ants = np.ones(N, int)
remaining_ants = m-N
for i in range(N):
    a = rd.randint(0,remaining_ants)
    ants[i] += a
    remaining_ants -= a

print(ants)