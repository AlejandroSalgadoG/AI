#pragma once

#include "Activations.h"

class NNet{
    int num_layers;
    int * layers;
    Activation ** activations;

    double ** W;
    double ** f; //structure to maintain forward information
    double * y;

    public:
        NNet(int * layers, int num_layers);
        ~NNet();

        void set_input(double * x);
        void set_weights(double * w, int layer);
        void set_activations(Activation * function, int layer);
        void set_labels(double * y);
        double* set_bias(double * x, int layer);
        double* get_weights(int layer);

        double* forward();
        double* dot_product(double * w, double * x, double * ans, int layer);
        double* activate(double * x, int layer);
};
