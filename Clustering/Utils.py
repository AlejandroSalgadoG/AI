import numpy as np

def calculate_dist_matrix( X, C, distance ):
    return np.array([ [ distance( x, c ) for c in C ] for x in X ])

def calculate_membership( X, C, distance ):
    D = calculate_dist_matrix( X, C, distance )
    return np.argmin( D, axis=1 )

def fuzzy_to_membership( U ):
    return np.argmax( U, axis=0 ), np.max( U, axis=0 )

def calculate_center( X, M ):
    return np.array([ X[ M == m ].mean( axis=0 ) for m in np.unique(M) ] )

def rescale_array( X, min_val, max_val ):
    return np.vstack( [ np.interp(x, [x.min(), x.max()], [min_val, max_val]) for x in X.T ] ).T
