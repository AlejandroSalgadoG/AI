import numpy as np

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
