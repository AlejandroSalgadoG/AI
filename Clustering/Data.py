import numpy as np
import matplotlib.image as matimg
from sklearn.manifold import TSNE

from Utils import rescale_array
from PlotResults import plot_data2d, plot_data3d, plot_data_as_img

img = matimg.imread("images/MNM_small.jpg").astype(int)
img_n, img_m,_ = img.shape
X = img.reshape( img_n*img_m, 3 )

#X = np.vstack( [ X.T, X.mean(axis=1), X.std(axis=1) ] ).T
#Xmax, Xmin = X.max(axis=0), X.min(axis=0)
#X = (X - Xmin ) / (Xmax - Xmin)
#
#X_embedded = TSNE(n_components=3, init="pca").fit_transform(X)
#X_embedded = rescale_array( X_embedded, 0, 255 )
#
#plot_data3d( X )
#plot_data3d( X_embedded )
#plot_data_as_img( img_n, img_m, X_embedded )
