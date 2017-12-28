#pragma once

class Activation{
    public:
        virtual double * activate(double * x, int size) = 0;
};

class Sigmoid: public Activation{
    public:
        double * activate(double * x, int size);
};
