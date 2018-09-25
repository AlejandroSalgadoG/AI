from Dataset import build_swiss_roll
from Plot3d import plot3d

from sklearn.neighbors import NearestNeighbors
import numpy as np

def calculate_distances(data, k):
    neighbors = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, neighbors = neighbors.kneighbors(data)
    return neighbors[:,1:]

def main():
    data, colors = build_swiss_roll(100)
    #plot3d(data, colors)

    neighbors = calculate_distances(data, 2)
    print(neighbors)

if __name__ == '__main__':
   main()
