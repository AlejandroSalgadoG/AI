#pragma once

#include "Activations.h"
#include "Loss.h"

class NNet{
    int num_layers;
    int * layers;
    Activation ** activations;
    Loss * loss_function;

    double alpha;
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
        void set_loss(Loss * loss_function);
        void set_learning_rate(double alpha);
        double* set_bias(double * x, int layer);
        double* get_weights(int layer);

        double* forward();
        double* dot_product(double * w, double * x, double * ans, int layer);
        double* activate(double * x, int layer);
        double loss(double * y_hat);
};
