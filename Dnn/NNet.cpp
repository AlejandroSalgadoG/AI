#include <math.h>
#include "NNet.h"

NNet::NNet(int * layers, int num_layers){
    this->layers = layers;
    this->num_layers = num_layers;

    W = new double*[num_layers-1]; //only the connections have weights
    f = new double*[num_layers]; //will contain the value of all neurons
    activations = new Activation*[num_layers]; //input layer has no activation function
                                               //so the first element is undefined
}

NNet::~NNet(){
    delete[] W;
    delete[] f;
    delete[] activations;
}

void NNet::set_input(double * x){
    this->f[0] = x;
}

void NNet::set_weights(double * w, int layer){
    W[layer] = w;
}

void NNet::set_activations(Activation * function, int layer){
    activations[layer] = function;
}

void NNet::set_labels(double * y){
    this->y = y;
}

void NNet::set_loss(Loss * loss_function){
    this->loss_function = loss_function;
}

void NNet::set_learning_rate(double alpha){
    this->alpha = alpha;
}

double* NNet::set_bias(double * x, int layer){
    x[layer] = 1;
    return x;
}

double* NNet::get_weights(int layer){
    return W[layer];
}

double* NNet::forward(){
    for(int i=1;i<num_layers;i++){       /*number of neurons plus bias*/
        f[i] = dot_product(W[i-1], f[i-1], new double[ layers[i] + 1 ], i);
        f[i] = activate(f[i], i);
        f[i] = set_bias(f[i], layers[i]);
    }

    return f[num_layers-1]; //return the output of the last layer
}

double* NNet::dot_product(double* w, double * x, double * ans, int layer){
    int num_neurons = layers[layer]; //number of neurons in the layer
    int num_inputs = layers[layer-1]+1; //number of inputs + bias

    for(int i=0;i<num_neurons;i++)
        for(int j=0;j<num_inputs;j++)
            ans[i] += w[i*num_inputs + j] * x[j];
    return ans;
}

double* NNet::activate(double * x, int layer){
    activations[layer]->activate(x, layers[layer]);
}

double NNet::loss(double * y_hat){
    return loss_function->calculate_loss(y, y_hat, layers[num_layers-1]);
}
