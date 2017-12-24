#pragma once

class NNet{
    int layers_size;
    int * layers;

    double ** W;
    double * x;
    double * y;

    public:
        NNet(int * layers, int layers_size);
        ~NNet();

        void set_input(double * x);
        void set_weights(double * w, int layer);
        double* get_weights(int layer);
        void set_labels(double * y);
};
