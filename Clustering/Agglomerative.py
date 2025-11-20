import numpy as np


def distance(x, y):
    return np.linalg.norm(np.subtract(x, y))


def calc_distance_clusters(clusters, distance_func):
    n = len(clusters)
    distances = np.zeros([n, n])
    for i in range(n):
        for j in range(i):
            dij = clusters[i].distance(clusters[j], distance_func)
            distances[i, j] = dij
            distances[j, i] = dij
    return distances


class Cluster:
    def __init__(self, X, indexes):
        assert X.ndim == 2  #  X must have to be matrix
        self.points = X
        self.center = np.mean(X, axis=0)
        self.indexes = indexes

    def distance(self, other_cluster, distance_func):
        return distance_func(self.center, other_cluster.center)

    def join(self, other_cluster):
        new_X = np.vstack([self.points, other_cluster.points])
        new_indexes = [*self.indexes, *other_cluster.indexes]
        return Cluster(new_X, new_indexes)


def select_clusters_to_join(distance_matrix):
    masked_distance = distance_matrix.copy()
    np.fill_diagonal(masked_distance, np.inf)
    return np.unravel_index(masked_distance.argmin(), masked_distance.shape)


def join_clusters(clusters, i, j):
    new_cluster = clusters[i].join(clusters[j])
    if i < j:
        clusters.pop(j)
        clusters.pop(i)
    else:
        clusters.pop(i)
        clusters.pop(j)
    clusters.append(new_cluster)
    return clusters


def update_clusters_distances(i, j, distance_clusters, clusters, distance_func):
    assert i != j
    # Delete rows-cols representing distances to clusters i and j
    n = len(distance_clusters)
    keep_index = [x for x in range(n) if x not in (i, j)]
    new_distances = distance_clusters.copy()
    new_distances = new_distances[keep_index]
    new_distances = new_distances[:, keep_index]
    # Add new row that represents distance from new cluster (last)
    # to the others
    new_cluster = clusters[-1]
    last_cluster_dists = [
        new_cluster.distance(o_cluster, distance_func) for o_cluster in clusters[:-1]
    ]
    # Add new col
    col_to_concat = np.array([last_cluster_dists]).T
    new_distances = np.hstack([new_distances, col_to_concat])
    row_to_concat = np.append(col_to_concat, 0).reshape(1, -1)
    new_distances = np.vstack([new_distances, row_to_concat])
    return new_distances


def agglomerative(X, distance_func, num_c=2):
    n = len(X)
    clusters = [Cluster(X[[i]], [i]) for i in range(n)]
    distance_clusters = calc_distance_clusters(clusters, distance_func)
    linkages = []
    while len(clusters) > num_c:
        i, j = select_clusters_to_join(distance_clusters)
        linkages.append([i, j])
        clusters = join_clusters(clusters, i, j)
        distance_clusters = update_clusters_distances(i, j, distance_clusters, clusters, distance_func)
    # Get labels of data original order
    labels = np.zeros((n, 1), dtype=int)
    for num_cluster, cluster in enumerate(clusters):
        indexes_clt = cluster.indexes
        labels[indexes_clt] = num_cluster
    return labels.flatten()
