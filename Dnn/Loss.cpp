#include <math.h>
#include "Loss.h"

double LessSquare::calculate_loss(double * y, double * y_hat, int size){
    double loss = 0;
    for(int i=0;i<size;i++)
        loss += pow(y[i] - y_hat[i], 2);
    return loss/2;
}
