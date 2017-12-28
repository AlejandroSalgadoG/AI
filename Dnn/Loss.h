#pragma once

class Loss{
    public:
        virtual double calculate_loss(double * y, double * y_hat, int size) = 0;
};

class LessSquare: public Loss{
    public:
        double calculate_loss(double * y, double * y_hat, int size);
};
