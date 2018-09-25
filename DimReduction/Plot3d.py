import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(dataset, colors):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    for idx, data in enumerate(dataset):
        x, y, z = data
        ax.scatter(x, y, z, color=colors[idx])

    plt.show()
