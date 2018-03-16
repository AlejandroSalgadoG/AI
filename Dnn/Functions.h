#pragma once

class Function{
    public:
        virtual void evaluate(double*, double*, int) = 0;
        virtual double derivative(double*, int) = 0;
        virtual char const* get_name() = 0;
};

class Sigmoid: public Function{
    public:
        void evaluate(double* x, double* ans, int size);
        double derivative(double* x, int element);
        char const* get_name();
};

class LessSquare: public Function{
    double * y;

    public:
        LessSquare(double * y);
        void evaluate(double* y_hat, double* ans, int size);
        double derivative(double* y_hat, int element);
        char const* get_name();
};