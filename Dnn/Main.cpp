#include <iostream>

#include "NNet.h"

using namespace std;

int main(int argc, char *argv[]){

    int size = 3;
    int structure[size] = {2,2,2};
    double input[2] = {0.05, 0.10}; 

    double weights_1[6] = { 
                            0.15, 0.25,
                            0.20, 0.30,
                            0.35, 0.35
                          };

    double weights_2[6] = {
                            0.40, 0.50,
                            0.45, 0.55,
                            0.60, 0.60
                          };

    double labels[2] = {0.01, 0.99};

    NNet * nnet = new NNet(structure, size);

    nnet->set_input(input);
    nnet->set_weights(weights_1, 1);
    nnet->set_weights(weights_2, 2);
    nnet->set_input(input);

    delete nnet;
}
