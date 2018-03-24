#pragma once

#include "Functions.h"

class NNet{
    int num_layers;
    int last_layer;
    int* layers;
    Activation** activations;
    Loss* loss_function;

    double alpha;
    double** W;
    double** f; //structure to maintain forward information
    double** b; //structure to maintain backward information

    public:
        NNet(int* layers, int num_layers);
        NNet(char const * file_name);
        ~NNet();

        void set_input(double* x);
        void set_weights(double* w, int layer);
        void set_activations(Activation* activation, int layer);
        void set_loss(Loss* loss);
        void set_labels(double* y);
        void set_learning_rate(double alpha);
        double* set_bias(double* x, int layer);
        double* get_weights(int layer);

        double* forward();
        double* dot_product(double* w, double* x, double* ans, int layer);
        double* activate(double* x, int layer);
        double loss(double* y_hat);
        void backward();
        void update_weights();

        void save(char const* file_name);
};
