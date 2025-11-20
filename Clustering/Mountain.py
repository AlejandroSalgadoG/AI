import numpy as np
from itertools import product


def density_function(v, x, param, distance_func):
    return np.exp(-distance_func(v, x) ** 2 / (2 * param ** 2))


def mountain_function(X, v, sigma, distance_func):
    return np.sum([density_function(v, x, sigma, distance_func) for x in X])


def create_grid(X, num_divisions):
    dimensions = np.linspace(X.max(axis=0), X.min(axis=0), num_divisions).T
    return list(product(*dimensions))


def calculate_mountain(X, V, sigma, distance_func):
    return [mountain_function(X, v, sigma, distance_func) for v in V]


def select_center(M, V):
    max_pos = np.argmax(M)
    return V[max_pos], M[max_pos]


def update_mountain(V, M, mc, c, sigma, beta, distance_func):
    for i, _ in enumerate(M):
        M[i] -= mc * density_function(V[i], c, beta, distance_func)
    return M


def calculate_membership(D):
    return np.argmin(D, axis=1)


def mountain(X, distance_func, num_c=2, num_div=1, sigma=0.1, beta=0.1):
    V = create_grid(X, num_div)
    M = calculate_mountain(X, V, sigma, distance_func)
    C = []
    for i in range(num_c):
        c, mc = select_center(M, V)
        M = update_mountain(V, M, mc, c, sigma, beta, distance_func)
        C.append(c)
    return np.array(C)
