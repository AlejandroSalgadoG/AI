import numpy as np

colors = [
           [31, 119, 180],
           [255, 127, 14],
           [44, 160, 44],
           [214, 39, 40],
           [148, 103, 189],
           [140, 86, 75],
           [227, 119, 194],
           [127, 127, 127],
           [188, 189, 34],
           [23, 190, 207]
         ]

def to_plt_color( color, alpha=1 ):
    return [ c / 255 for c in color ] + [alpha]

def get_plt_color( i, alpha=1 ):
    color = colors[ i % 10 ]
    return [ c / 255 for c in color ] + [alpha]

def get_color( i ):
    return colors[ i % 10 ]
