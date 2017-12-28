#include <math.h>
#include "Activations.h"

double* Sigmoid::activate(double * x, int size){
    for(int i=0;i<size;i++)
        x[i] = 1/(1+exp(x[i] * -1));
    return x;
} 
