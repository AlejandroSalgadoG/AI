import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.image import imread


face = imread("face.png")
eye = imread("eye.png")

a = face[:, :, 0]
w = eye[:, :, 0]

plt.imshow(a, cmap='gray', vmin=0, vmax=1)
plt.show()

plt.imshow(w, cmap='gray', vmin=0, vmax=1)
plt.show()

conv = convolve2d(a, np.fliplr(np.flipud(w)), mode="valid")

plt.imshow(conv, cmap='gray')
plt.show()
