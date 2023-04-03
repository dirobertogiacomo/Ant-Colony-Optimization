import numpy as np

distance = np.array([[999,3,6,2,3],[3,999,5,2,3],[6,5,999,6,4],[2,2,6,999,6],[3,3,4,6,999]])
delta_t_k = np.zeros((5,5))

city = 0

length = 0

path = np.array([3,1,4,2,0])

j = city
for i in np.nditer(path):

    length += distance[j, i]
    delta_t_k[j, i] = 1
    delta_t_k[i, j] = 1
    
    j=i

print(length)
print(delta_t_k)