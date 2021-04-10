import numpy as np

# Ejemplo de una red neuronal simple
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

from Functions import sigmoid, d_sigmoid, mse, d_mse, add_bias # importar las funciones con sus respectivas derivadas
                                                               # add_bias esta explicado en el archivo de funciones
class NNet:
    # Este es el constructor de la clase, recibe una lista que contenga el numero
    # de neuronas que se quieren en cada capa sin tener en cuenta el bias.
    # Es decir, para una red que tenga 3 entradas, 2 capas ocultas con 5 y 4
    # neuronas respectivamente y 1 sola salida, arch tendria que ser [3, 5, 4, 1]
    def __init__(self, arch):
        self.arch = arch
        self.n_layers = len(arch)-1 # se le tiene que restar 1 porque las listas
                                    # empiezan en 0, entonces si la red neuronal
                                    # tiene entrada, 2 capas y salida, la lista
                                    # de arquitectura tendria 4 posiciones,
                                    # de 0 a 3

    # Esta funcion hace el papel de forward, recive la entrada de la red (data)
    # y los pesos de cada capa. El arreglo de los pesos debe tener el mismo
    # numero de elementos que la lista de arquitectura (arch) menos uno. Esto
    # debido a que los pesos conectan las capas. Es decir, si la red es la del
    # ejemplo, entonces existe una posicion que representa los pesos de la
    # conexion de la entrada a la primera capa oculta, y otra que representa
    # los pesos de la capa ocula a la salida.

    # el formato de data es simplemente una lista que contanga una posicion por
    # cada entrada a la red.

    # el formato de los pesos es: en las filas van los pesos que estan relacionados
    # con las neuronas de las que sale la informacion y en las columnas estan
    # a las neuronas a las que llegan. Es decir, el primer indice de la lista
    # de pesos indica con cual conjunto de pesos se va a trabajar, weights[0]
    # es una matriz que tiene los pesos que conectan la entrada con la primera
    # capa oculta. Luego en esa matriz la posicion 0,2 siginifica el peso que va
    # de la entrada 0 a la neurona 2 de la capa oculta (recuerda que los indices
    # empiezan en 0, entonces la neurona de la posicion 2 es la tercera de esa
    # capa). Ese peso se accede como weights[0][0,2]

    def predict(self, data, weights):
        for layer in range( self.n_layers ): # por cada capa (0,1,2,...)
            data = np.append(data, 1) # agrega el bias a la entrada de la capa
            w_data = np.matmul(data, weights[layer]) # pondera las entradas por los pesos respectivos
            data = sigmoid(w_data) # y en cada neurona de la capa siguiente aplica una funcion sigmoide
        return data # retorna la salida de la red

    # Esta funcion hace lo mismo de la anterior, la unica diferencia es que
    # guarda la informacion que las neuronas de cada capa van sacando para no
    # tener que recalcular esos valores durante el entrenamiento de la red
    # (parte backward del algoritmo)
    def test_prediction(self, data, weights):
        inter_data = [] # aqui se va a guardar la salida de cada capa
        for layer in range( self.n_layers ):
            inter_data.append(data) # guarda la salida de las capas, (la primera posicion va a ser la entrada de la red como tal)
            data = np.append(data, 1) # hace lo mismo del metodo anterior
            w_data = np.matmul(data, weights[layer])
            data = sigmoid(w_data)
        inter_data.append(data) # guarda la salida final de la red
        return data, inter_data # retorna ambos valores como una tupla

    # Esta es la funcion que tiene el codigo del metodo backward del algoritmo,
    # los parametros data y weighs tienen las mismas dimensiones que en el
    # metodo predict, contienen la entrada de la red y el valor inicial de los
    # pesos con los que se va a empezar a entrenar la red. El parametro label
    # es la prediccion que quiero que la red haga cuando se determinen los valores
    # de data como entrada. Por ultimo, eta es un numero que indica la tasa
    # de aprendizaje de la red
    def train(self, data, label, weights, eta):
        updates = np.zeros_like(weights) # crea un arreglo con las mismas dimensiones
                                         # de los pesos para guardar las actualizaciones
                                         # que se deben hacer en cada paso

        y_bar, inter_data = self.test_prediction(data, weights) # Aqui se puede ver que resultado esta sacando la red con la
                                                                # entrada indicada en data y los pesos definidos en weights

        error = mse(label, y_bar) # Calcula el error total, este valor es solo informativo, no se usa en el algorimo,
                                  # se puede imprimir para ver como va mejorando la prediccion de la red a medida
                                  # que el algoritmo avanza

        d_error = d_mse(label, y_bar) # calcula la derivada del error, se hace fuera del ciclo porque solo la primera
                                      # vez se calcula respecto al error cuadratico medio. Esto se puede ver como dE/dO
                                      # que es la derivada del error respecto a la salida, tiene dimensiones 1xN, donde
                                      # N es el numero de salidas. Se calcula como d/dO 1/2 (Y-O)^2 y da (Y-O)*-1, osea
                                      # (O-Y)

        for b_layer in range(self.n_layers,0,-1):  # por cada capa desde la ultima hasta la primera
            d_out = d_sigmoid(inter_data[b_layer]) # Se calcula la derivada de lo que salio de la neura con respecto a lo que entro
                                                   # es decir dO/dH, que es d/dH 1/(1-e^-H) y da O-(1-O), esto da una matriz de
                                                   # 1xN.

            d_in = add_bias(inter_data[b_layer-1]) # Aqui se agrega el bias a la salida de la capa anterior (por eso b_layer-1)

            d_w = np.matmul(d_in.T, d_error*d_out) # ahora se calcula la derivada del error con respecto a lo que le entro a las
                                                   # neuronas, es decir dE/dH como dE/dO * dO/dH, como es lo que entro a la
                                                   # neurona, no importan las neuronas que existan en la capa anterior, simplemente
                                                   # se necesita lo que entro finalmente, lo que resulto despues de multiplicar los
                                                   # pesos, por lo que este vector tambien va a ser de 1*N
                                                   # luego se calcula dH/dW, osea d/dW O_1*w_1+ O_2*w_2 + ... , lo que da O_1, O_2, ...
                                                   # donde las O son las salidas de la capa anterior, es decir lo que se calculo
                                                   # en d_in. Finalmente calculas el cambio del error respecto a los pesos, es decir,
                                                   # dE/dW como la multiplicacion de esos 2 vectores, el resultado es de M*N, donde
                                                   # N es el numero de neuronas que tiene la capa actual (Sin tener en cuenta el bias)
                                                   # y M es el numero de neuronas de la capa anterior

            updates[b_layer-1] = d_w*eta # Solo queda guardar cada dE/dW multiplicado por la tasa de aprendizaje en la matriz que
                                         # va a contener las actualizaciones

            d_error = np.matmul(weights[b_layer-1][:-1], (d_error*d_out).T).T # Hasta la linea anterior va una iteracion completa
                                                                              # del algoritmo, pero como el error a partir de la ultima
                                                                              # capa oculta ya no se calcula con base al error cuadratico
                                                                              # medio sino con respecto a las sigmoides de la capa siguiente
                                                                              # se calcula el error de la capa siguiente antes de emepezar
                                                                              # su iteracion
            # El error de los pesos que conectan a una capa diferente a la de salida se calcula teniendo en cuenta que cada neurona de la capa
            # anterior esta conectada a todas las neuronas de la capa que presede, entonces el error debe calcularse como
            # dE/dO = dE/dO_1 + dE/dO_2+... , donde la O sin indice representa el error que se va a utilizar en la siguiente iteracion, y la O
            # indexada como el error que se genera en cada neurona de la capa siguiente, es decir

            # O -- O_1
            #  \
            #   \
            #    \ O_2

            # se quiere calcular el la influencia de lo que salio de la neurona O, pero esa neurona esta conectada a 2 neuronas de la capa
            # siguiente, entonces se calcula el error de las neuronas de la capa siguiente dE/O_1 y dE/dO_2 y se suman para calcular el error
            # de la neurona O. Ahora dE/dO_1 y dE/dO_2 es lo que se calculo en la iteracion como d_error*d_out. Por ultimo el bias no influye
            # en la propagacion de error de una capa anterior, entonces no es tenido en cuenta, por eso se accede a weights[b_layer-1] (pesos
            # de la capa anterior) y se le quitan los del bias, weights[b_layer-1][:-1] (El -1 significa la ultima posicion, es decir :-1 es todo
            # sin la ultima posicion). Las transpuestas son para que la operacion matricial sea valida y el vector resultante sea de dimension
            # 1xN como el primer vector de error

        return error, weights-updates # retorna el error calculado y los pesos actualizados

    def init_random_weights(self, low=0, high=1):
        return np.array([ np.random.uniform( low, high, size=(self.arch[i]+1, self.arch[i+1]) ) for i in range(self.n_layers) ])
