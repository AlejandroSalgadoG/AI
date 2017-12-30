#include <iostream>
#include "NNet.h"

using namespace std;

int main(int argc, char *argv[]){

    int num_layers = 3;
    int structure[num_layers] = {2,2,2}; //first layer are inputs

    double input[3] = {0.05, 0.10, 1}; //two inputs plus bias

    double weights_1[6] = { 0.15, 0.20, 0.35,
                            0.25, 0.30, 0.35 };

    double weights_2[6] = { 0.40, 0.45, 0.60,
                            0.50, 0.55, 0.60 };

    double labels[2] = {0.01, 0.99};

    NNet * nnet = new NNet(structure, num_layers);

    nnet->set_input(input);
    nnet->set_weights(weights_1, 0);
    nnet->set_activations(new Sigmoid(), 1);
    nnet->set_weights(weights_2, 1);
    nnet->set_activations(new Sigmoid(), 2);
    nnet->set_labels(labels);
    nnet->set_loss(new LessSquare());
    nnet->set_learning_rate(0.5);

    double * ans = nnet->forward();
    double loss = nnet->loss(ans);

    cout << "loss " << loss << endl;

    delete nnet;
}
