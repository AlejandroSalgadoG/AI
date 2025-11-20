import numpy as np


def select_centers( n, num_c ):
    return np.random.choice( n, size=num_c, replace=False )

def calc_dist_matrix( X, C, distance_func):
    return np.array([ [ distance_func( x, c ) for c in C ] for x in X ])

def calc_membership(D):
    return np.argmin( D, axis=1 )

def cost_function(D, M):
    I = np.arange( len(D) )
    return np.sum( D[I, M] )

def update_centers(X, C, M):
    for i,_ in enumerate(C):
        X_c = X[ M == i ]
        C[i] = np.sum( X_c, axis=0 ) / len( X_c )
    return C

def kmeans( X, distance_func, num_c=2, iters=1, print_error=True):
    n,_ = X.shape
    C = X[ select_centers( n, num_c ) ]
    for ite in range(iters):
        D = calc_dist_matrix( X, C, distance_func)
        M = calc_membership( D )
        if print_error: print( cost_function( D, M ) )
        C = update_centers( X, C, M )
    return C, M
