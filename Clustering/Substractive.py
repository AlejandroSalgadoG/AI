import numpy as np
from itertools import product


def density_function(x, v, denominator, distance_func):
    return np.exp(-distance_func(x, v) ** 2 / denominator)


def substraction_function(X, v, denominator, distance_func):
    return np.sum([density_function(x, v, denominator, distance_func) for x in X])


def calculate_substraction_vector(X, ra, distance_func):
    return [substraction_function(X, x, denominator=(ra / 2) ** 2, distance_func= distance_func) for x in X]


def select_first_center(M, V):
    return V[np.argmax(M)]


def update_substraction(X, M, c, rb, distance_func):
    M_c = np.max(M)
    for i, _ in enumerate(M):
        M[i] -= M_c * density_function(X[i], c, denominator=(rb / 2) ** 2, distance_func=distance_func)
    return M


def calculate_membership(D):
    return np.argmin(D, axis=1)


def substract(X, distance_func, num_c=2, num_div=1, ra=1.0, rb=None):
    if not rb:
        rb = 1.5 * ra
    M = calculate_substraction_vector(X, ra, distance_func)
    C = []
    for i in range(num_c):
        c = select_first_center(M, X)
        M = update_substraction(X, M, c, rb, distance_func)
        C.append(c)
    return np.array(C)
