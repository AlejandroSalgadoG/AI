#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "NNet.h"

using namespace std;

NNet::NNet(int * layers, int num_layers){
    this->layers = layers;
    this->num_layers = num_layers;
    last_layer = num_layers-1;

    W = new double*[num_layers-1]; //only the connections have weights
    b = new double*[num_layers-1]; //will contain weights updates
    f = new double*[num_layers]; //will contain the value of all neurons

    activations = new Activation*[num_layers]; //will contain activations, the first layer
                                               //are inputs without activation,
                                               //so the first possition is null
}

NNet::NNet(char const * file_name){
    ifstream net_file;
    istringstream reader;
    string line;
    int input_layer = 0;

    net_file.open(file_name);

    getline(net_file, line);
    reader.str(line);
    reader >> num_layers;
    reader.clear();

    layers = new int[num_layers];

    getline(net_file, line);
    reader.str(line);
    for(int i=0;i<num_layers;i++)
        reader >> layers[i];
    reader.clear();

    last_layer = num_layers-1;

    W = new double*[num_layers-1];
    b = new double*[num_layers-1];
    f = new double*[num_layers];
    activations = new Activation*[num_layers];

    getline(net_file, line);
    reader.str(line);
    for(int layer=input_layer;layer<last_layer;layer++){
        int num_neur_next = layers[layer+1]; //height of the matrix
        int num_neur_actual = layers[layer]+1; //width of the matrix

        double * w = new double[num_neur_actual*num_neur_next];
        for(int i=0;i<num_neur_next;i++)
            for(int j=0;j<num_neur_actual;j++)
                reader >> w[i*num_neur_actual + j];
        W[layer] = w;
    }
    reader.clear();

    string function;
    getline(net_file, line);
    reader.str(line);
    for(int layer=input_layer;layer<last_layer+1;layer++){
        reader >> function;
        cout << function << endl;
    }

    net_file.close();
}

NNet::~NNet(){
    delete[] W;
    delete[] b;
    delete[] f;
    delete[] activations;
    delete loss_function;
}

void NNet::set_input(double * x){
    this->f[0] = x;
}

void NNet::set_weights(double * w, int layer){
    W[layer] = w;
}

void NNet::set_activations(Activation * activation, int layer){
    activations[layer] = activation;
}

void NNet::set_labels(double * y){
    this->y = y;
}

void NNet::set_loss(Loss * loss){
    loss_function = loss;
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

    return f[last_layer]; //return the output of the last layer
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
    activations[layer]->evaluate(x, x, layers[layer]);
    return x;
}

double NNet::loss(double * y_hat){
    double loss;
    loss_function->evaluate(y_hat, &loss, layers[last_layer]);
    return loss;
}

void NNet::backward(){

    double dE_dy, dy_do, dE_do, do_dz, do_dw, dE_dw;
    double* w;

    int num_neur_actual, num_neur_next, num_neur_past;
    int con_layer, bias_pos;

    for(int layer=last_layer;layer>0;layer--){ //for each layer

        num_neur_actual = layers[layer]; //height of the matrix
        num_neur_next = layers[layer-1]+1; //width of the matrix

        con_layer = layer-1; //index for W and b
        bias_pos = num_neur_next-1; //bias possition in the matrix of weights

        b[con_layer] = new double[num_neur_actual * num_neur_next]; //allocate space for backward information

        for(int i=0;i<num_neur_actual;i++){ //for each neuron

            if(layer == last_layer) dE_dy = loss_function->derivative(f[layer], i); //deriv error respect to final output
            else{
                dE_dy = 0; //reset the deriv error respect to the output
                num_neur_past = layers[layer+1]; //number of forward paths, bias is not a forward pass

                for(int j=0;j<num_neur_past;j++){ //for each forward path
                    dE_do = b[con_layer+1][j*num_neur_next + bias_pos]; //dE_do is the change of the bias in the next connection layer
                    do_dz = W[con_layer+1][j*num_neur_next + i];
                    dE_dy += dE_do * do_dz; //add the change of the error respect to each path
                }
            }

            dy_do = activations[layer]->derivative(f[layer], i); //deriv output respect to activation
            dE_do = dE_dy * dy_do; //deriv error respect to activation

            b[con_layer][i*num_neur_next + bias_pos] = dE_do; //set bias update

            for(int j=0;j<num_neur_next-1;j++){ //for each weight except the bias

                do_dw = f[layer-1][j]; //deriv activation respect to weight
                dE_dw = dE_do * do_dw; //deriv error respect to weight

                b[con_layer][i*num_neur_next + j] = dE_dw; //set weight update
            }
        }
    }
}

void NNet::update_weights(){

    double* w;
    double dE_dw;

    int num_neur_actual, num_neur_next;

    for(int layer=last_layer;layer>0;layer--){ //for each layer

        num_neur_actual = layers[layer]; //height of the matrix
        num_neur_next = layers[layer-1]+1; //width of the matrix

        for(int i=0;i<num_neur_actual;i++){ //for each neuron
            for(int j=0;j<num_neur_next;j++){ //for each weight

                dE_dw = b[layer-1][i*num_neur_next + j]; //get weight update
                w = &W[layer-1][i*num_neur_next + j]; //get position of weight
                *w = *w - alpha*dE_dw; //update weight
            }
        }
    }

}

void NNet::save(char const * file_name){
    ofstream net_file;
    int num_neur_actual, num_neur_next;
    int input_layer = 0;

    net_file.open(file_name);

    net_file << " " << num_layers << endl;

    for(int i=input_layer;i<num_layers;i++) // for all layers
        net_file << " " << layers[i];
    net_file << endl;

    for(int layer=input_layer;layer<last_layer;layer++){ //for each layer

        num_neur_next = layers[layer+1]; //height of the matrix
        num_neur_actual = layers[layer]+1; //width of the matrix

        for(int i=0;i<num_neur_next;i++) //for each neuron in the next layer
            for(int j=0;j<num_neur_actual;j++) //for each neuron in the current layer
                net_file << " " << W[layer][i*num_neur_actual + j]; //write weight
    }
    net_file << endl;

    for(int i=input_layer+1;i<=num_layers;i++) //for all activation functions
        net_file << " " << activations[i]->get_name();
    net_file << endl;

    net_file.close();
}
