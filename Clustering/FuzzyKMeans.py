import numpy as np


def initialize_membership(n, num_c):
    U = np.random.uniform( size=( num_c, n ) )
    U /= np.sum(U, axis=0)
    return U

def calculate_centers(X, Um):
    C = np.array([ np.sum( X * u_m.reshape(-1,1), axis=0 ) for u_m in Um ])
    C /= np.sum( Um, axis=1 ).reshape(-1,1)
    return C

def calculate_distances( X, C, distance_func ):
    return np.array( [ [ distance_func( x, c ) for x in X ] for c in C ] )
        
def cost_function( Um, D ):
    return np.sum( Um * D**2 )

def update_membership( D, m ):
    return 1 / sum( [ (D / d) ** (2 / (m-1)) for d in D ] )
        
def fuzzy_kmeans(X, distance_func, m=2, num_c=2, iters=1, print_error=True):
    n,_ = X.shape
    U = initialize_membership(n, num_c)
    for ite in range(iters):
        Um = U ** m
        C = calculate_centers( X, Um )
        D = calculate_distances( X, C, distance_func)
        if print_error: print( cost_function( Um, D ) )
        U = update_membership( D, m )
    return C, U
