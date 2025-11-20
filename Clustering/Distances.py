import numpy as np

def euclidean(x, y):
    return np.linalg.norm( np.subtract( x, y ) )
