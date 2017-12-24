#include "NNet.h"

NNet::NNet(int * layers, int layers_size){
    this->layers = layers;
    this->layers_size = layers_size;

    W = new double*[layers_size-1];
}

NNet::~NNet(){
    delete[] W;
}

void NNet::set_input(double * x){
    this->x = x;
}

void NNet::set_weights(double * w, int layer){
    W[layer] = w;
}

double* NNet::get_weights(int layer){
    return W[layer];
}

void NNet::set_labels(double * y){
    this->y = y;
}
