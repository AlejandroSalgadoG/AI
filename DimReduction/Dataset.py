import numpy as np
from sklearn import datasets

def build_swiss_roll(samples):
    data, _ = datasets.make_swiss_roll(samples)
    data = np.array(data)

    colors = np.chararray(samples)

    vertical = data[:,2]
    horizontal = data[:,0]

    red = np.where((vertical >= 0) & (horizontal > 0))
    green = np.where((vertical < 0) & (horizontal >= 0))
    blue = np.where((vertical <= 0) & (horizontal < 0))
    yellow = np.where((vertical > 0) & (horizontal <= 0))

    colors[red] = "red"
    colors[green] = "green"
    colors[blue] = "blue"
    colors[yellow] = "yellow"

    return data, colors
