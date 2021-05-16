import numpy as np
import pandas as pd
np.random.seed(1234567)

from NNet import NNet
from Functions import *

from sklearn.model_selection import train_test_split

def train( epoch, tol, arch, eta, x, y, batch_size=100 ):
    print("arch", arch)
    print("eta", eta)
    print("batch", batch_size)

    nnet = NNet(arch)
    weights = nnet.init_random_weights()

    n = x.shape[0]
    errors = np.zeros(n)

    for i in range( epoch ):
        batch_x, batch_y = get_batch( x, y, batch_size )
        for idx, (input_data, label) in enumerate(zip(batch_x, batch_y)): 
            errors[idx], weights = nnet.train(input_data, label, weights, eta)
        mean_error = errors.mean()
        if mean_error < tol: break
        print(i, "%.8f" % mean_error, end="\r" )
    print()

    return nnet, weights

dataset = pd.read_csv("heart.csv")
x = dataset[ ["age", "trtbps", "chol", "thalachh", "oldpeak"] ]
y = dataset[ "output" ]
x,y = normalize( x.values ), classes2binary( y.values )

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, shuffle=True )
nnet, weights = train( epoch=500, tol=1e-2, arch=[5, 5, 5, 2], eta=0.2, x=x_train, y=y_train )
    
y_pred = predict( nnet, weights, x_test )
y_true = binary2class( y_test )
plot_result( y_true, y_pred )
