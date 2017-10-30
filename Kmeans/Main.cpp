#include <iostream>
#include <cuda_runtime.h>

#include "Kmeans.h"

using namespace std;

int main(int argc, char *argv[]){

    int features = 20;
    int samples = 20;

    int size = features * samples;

    srand(1);

    float * h_samples = new float[size];
    for(int i=0;i<size;i++)
        h_samples[i] = (rand() % 100);

    for(int i=0;i<samples;i++){
    	for(int j=0;j<features;j++)
			cout << h_samples[j+i*20] << " ";
		cout << endl;
	}

    float * d_samples;
    cudaMalloc(&d_samples, sizeof(float) * size);
	cudaMemcpy(d_samples, h_samples, sizeof(float) * size, cudaMemcpyHostToDevice);

	cout << endl << "Starting kernel...";
	kmeans(d_samples, samples, features);
	cout << "done" << endl << endl;

	cudaMemcpy(h_samples, d_samples, sizeof(float) * size, cudaMemcpyDeviceToHost);

    for(int i=0;i<samples;i++){
    	for(int j=0;j<features;j++)
			cout << h_samples[j+i*20] << " ";
		cout << endl;
	}

    cudaFree(d_samples);

    return 0;
}
