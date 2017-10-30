#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#include "CudaUtilities.h"

using namespace std;

__device__
float euclidian_distance(float * sample, float * centroid, int features){
    float result = 0;
    for(int j=0;j<features;j++) result += pow(sample[j] - centroid[j], 2);
    return sqrt(result);
}

__global__
void centroid_calculation(float * d_samples, float * d_centroids, int * d_class, int samples, int features, int k){
    int sample_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(sample_idx >= samples) return;

    float * sample = &d_samples[sample_idx*features];
    float * centroid;

    int classification = 0;
    float result, best_result;

    for(int i=0;i<k;i++){
        centroid = &d_centroids[i*features];

        result = euclidian_distance(sample, centroid, features);

        if(i == 0) best_result = result;
        else if(result < best_result){
            classification = i;
            best_result = result;
        }
    }

    d_class[sample_idx] = classification;
}

__global__
void compare_classes(int * d_class, int * d_past_class, int samples, bool * are_equal){
    int sample_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(sample_idx >= samples) return;

    if(d_class[sample_idx] != d_past_class[sample_idx]) *are_equal = false;
}

__global__
void centroid_movement(float * d_samples, float * d_centroids, int * d_class, int samples, int features, int k){
    int kfeature_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(kfeature_idx >= features*k) return;

    int feat = threadIdx.x % features;
    int centroid = kfeature_idx / features;

    float numerator=0, denominator=0;

    for(int i=0;i<samples;i++){
        if(d_class[i] == centroid){
            numerator += d_samples[feat + i*features];
            denominator += 1;
        }
    }

    d_centroids[feat + centroid*features] = numerator/denominator;
}

int* kmeans(float * h_samples, float * h_centroids, int * h_class, int samples, int features, int k, int max_iterations){
    int samples_size = features * samples;
    int centroids_size = features * k;

    int * d_class;
    int * d_past_class;
    float * d_samples;
    float * d_centroids;

    cudaMalloc(&d_class, sizeof(int) * samples);
    cudaMalloc(&d_past_class, sizeof(int) * samples);
    cudaMalloc(&d_samples, sizeof(float) * samples_size);
    cudaMalloc(&d_centroids, sizeof(float) * centroids_size);

    cudaMemcpy(d_past_class, h_class, sizeof(float) * samples, cudaMemcpyHostToDevice);
    cudaMemcpy(d_samples, h_samples, sizeof(float) * samples_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, sizeof(float) * centroids_size, cudaMemcpyHostToDevice);

    cudaDeviceProp device_prop = get_device_properties();
    kernel_dim dist_kernel = get_calculation_kernel_dimensions(device_prop, samples);
    kernel_dim cent_kernel = get_movement_kernel_dimensions(device_prop, features, k);

    int iteration = 0;

    bool *d_are_equal;
    bool h_are_equal = false;
    bool h_d_are_equal_initilizer = true;

    cudaMalloc(&d_are_equal, sizeof(bool));

    cout << endl << "Starting kmeans" << endl << endl;
    while(!h_are_equal && iteration++ < max_iterations){
        cout << "iteration " << iteration << endl;

        centroid_calculation<<<dist_kernel.blk_size, dist_kernel.thr_size>>>(d_samples, d_centroids, d_class, samples, features, k);

        cudaMemcpy(d_are_equal, &h_d_are_equal_initilizer, sizeof(bool), cudaMemcpyHostToDevice);
        compare_classes<<<dist_kernel.blk_size, dist_kernel.thr_size>>>(d_class, d_past_class, samples, d_are_equal);
        cudaMemcpy(&h_are_equal, d_are_equal, sizeof(bool), cudaMemcpyDeviceToHost);

        if(!h_are_equal){
            cudaMemcpy(d_past_class, d_class, sizeof(float)*samples, cudaMemcpyDeviceToDevice);

            centroid_movement<<<cent_kernel.blk_size, cent_kernel.thr_size>>>(d_samples, d_centroids, d_class, samples, features, k);
        }
    }
    cout << endl << "done" << endl << endl;

    cudaMemcpy(h_class, d_class, sizeof(int)*samples, cudaMemcpyDeviceToHost);

    cudaFree(d_class);
    cudaFree(d_centroids);
    cudaFree(d_samples);

    return h_class;
}
