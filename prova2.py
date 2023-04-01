import numpy as np

A = np.arange(25).reshape(5,5)
B = np.zeros((5,5))
B[0,0] = 1
p = 0.5

print(A)

A = (1-p)*A + B
print(A)
