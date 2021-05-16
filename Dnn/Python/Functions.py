import numpy as np
import matplotlib.pyplot as plt

def add_bias(x):
    ans = np.append(x,1) # agregar el bias al vector de entrada
    return np.expand_dims(ans, 0) # esta linea es por una cosa tecnica de numpy,
                                  # para esa libreria, un arreglo no es un vector
                                  # propiamente. Es solo una lista de cosas, y su
                                  # shape es (n,), entonces si quieres hacer
                                  # operaciones matriciales, es recomendable que
                                  # lo combiertas a vector, en numpy es agregarle
                                  # una dimension, y eso es lo que hace esta
                                  # linea, añade la dimension que falta para que
                                  # el arreglo se pueda ver como un vector con
                                  # dimension (n,1)
    # Esta misma operacion se tiene que realizar en todas las funciones de
    # derivada para que el metodo que hace el backward pueda hacer las operaciones
    # de matrices bien.

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    ans = x*(1-x)
    return np.expand_dims(ans, 0)  

def mse(y, y_bar):
    return np.sum((y-y_bar)**2/2)

def d_mse(y, y_bar):
    ans = y_bar-y
    return np.expand_dims(ans, 0)

def normalize( x ):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def classes2binary( y ):
    classes = np.unique(y)
    bin_classes = np.zeros( shape=( y.size, classes.size ) )
    for i, c in enumerate(y): bin_classes[i,c] = 1
    return bin_classes

def binary2class( y ):
    n,_ = y.shape
    classes = np.zeros( n )
    for i, c in enumerate(y): classes[i] = np.argmax(c)
    return classes

def hard_classification( y_hat, threshold=0.5 ):
    y_hard = np.zeros( y_hat.shape )
    y_hard[ y_hat > threshold ] = 1
    return y_hard

def predict( nnet, weights, x ):
    y_hat = np.array( [ nnet.predict(input_data, weights) for input_data in x ] )
    y_hard = hard_classification( y_hat, threshold=0.5 )
    y_pred = binary2class( y_hard )
    return y_pred

def plot_result( y_true, y_pred ):
    for i, (true,pred) in enumerate(zip(y_true,y_pred)):
        plt.hlines( true+0.025, i, i+0.9, color="r", linewidth=10 )
        plt.hlines( pred-0.025, i, i+0.9, color="b", linewidth=10 )
    plt.yticks([0,1,1.2], ["0", "1", ""])
    plt.legend(["true", "predicted"])
    plt.show()

def get_batch(x, y, batch_size):
    idx = np.random.choice( np.arange(batch_size), size=batch_size, replace=False )
    return x[idx], y[idx]
