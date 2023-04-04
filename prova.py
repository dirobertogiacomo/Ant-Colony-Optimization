import numpy as np
import matplotlib.pyplot as plt

distance = np.array([[999,3,6,2,3],[3,999,5,2,3],[6,5,999,6,4],[2,2,6,999,6],[3,3,4,6,999]])

TYPE_MATRIX = ['EXPLICIT', 'EUC_2D']

pluto = 'EXPLICIT'

if pluto in TYPE_MATRIX:
    print('ciao')
else:
    print('cacca')