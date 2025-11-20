from Data import *
from Utils import *
from Distances import *
from PlotResults import *

from Kmeans import kmeans
from FuzzyKMeans import fuzzy_kmeans
from Mountain import mountain
from Substractive import substract
from Agglomerative import agglomerative

C, M = kmeans( X, euclidean, num_c=7, iters=30 )
#C_embedded = calculate_center( X_embedded, M )
#plot_result2d( X, M, C )
plot3d( X, M, C )
plot_result_as_img( img_n, img_m, X, M, C )

#C, U = fuzzy_kmeans( X, euclidean, num_c=7, iters=30 )
#M, Umax = fuzzy_to_membership( U )
#C_embedded = calculate_center( X_embedded, M )
#plot_result2d( X, M, C, Umax )
#plot3d( X, M, C, Umax )
#plot_result_as_img( img_n, img_m, X, M, C, Umax )

#C = mountain( X, euclidean, num_c=7, num_div=5)
#M = calculate_membership( X, C, euclidean )
#C_embedded = calculate_center( X_embedded, M )
#plot_result2d( X, M, C)
#plot3d( X, M, C)
#plot_result_as_img( img_n, img_m, X, M, C )

#C = substract( X, euclidean, num_c=7, num_div=5)
#M = calculate_membership( X, C, euclidean )
#C_embedded = calculate_center( X_embedded, M )
#plot_result2d( X, M, C )
#plot3d( X, M, C )
#plot_result_as_img( img_n, img_m, X, M, C)

#M = agglomerative(X, euclidean, num_c=7 )
#C = calculate_center( X, M )
#C_embedded = calculate_center( X_embedded, M )
#plot_result2d( X, M, C )
#plot3d( X, M, C )
#plot_result_as_img( img_n, img_m, X, M, C )
