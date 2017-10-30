#include <iostream>
#include <sstream>
#include <cstring>

using namespace std;

void get_parameters(int * samples, int * features, int * k, int * max_iterations){
    cin >> *samples;
    cin >> *features;
    cin >> *k;
    cin >> *max_iterations;
    cin.ignore(); //ignore new line
}

float * get_samples(int samples, int features){
    float * h_samples = new float[samples * features];

    string input;
    int data;
    for(int i=0;cin;i++){
        getline(cin, input);
        istringstream sample(input);
        for(int j=0;sample >> data;j++) h_samples[j+i*features] = data;
    }

    return h_samples;
}

float* initialize_centroids(float * h_samples, int k, int features){
    float * h_centroids = new float[k*features];

    float *centroid, *sample;

    for(int i=0;i<k;i++){
        centroid = &h_centroids[i*features];
        sample = &h_samples[i*features];

        memcpy(centroid, sample, sizeof(float) *features);
    }

    return h_centroids;
}

void print_samples(float * h_samples, int features, int samples){
    for(int i=0;i<samples;i++){
        for(int j=0;j<features;j++)
            cout << h_samples[j+i*features] << " ";
        cout << endl;
    }
}

void print_labeled_samples(float * h_samples, int * h_class, int features, int samples){
    for(int i=0;i<samples;i++){
        for(int j=0;j<features;j++)
            cout << h_samples[j+i*features] << " ";
        cout << h_class[i] << endl;
    }
}
