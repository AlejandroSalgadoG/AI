#include <iostream>
#include <cstring>

using namespace std;

float* initialize_rand_samples(int seed, int range, int size){
    srand(seed);

    float * h_samples = new float[size];
    for(int i=0;i<size;i++) h_samples[i] = (rand() % range);
    return h_samples;
}

float* initialize_rand_centroids(int seed, float * h_samples, int k, int samples, int features){
    srand(seed);

    float * h_centroids = new float[k*features];

    int random;
    float *centroid, *sample;

    for(int i=0;i<k;i++){
        random = rand() % samples;

        centroid = &h_centroids[i*features];
        sample = &h_samples[random*features];

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
