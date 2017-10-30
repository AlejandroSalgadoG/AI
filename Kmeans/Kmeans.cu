#include <iostream>
#include <cuda_runtime.h>

using namespace std;

struct kernel_dim{
    dim3 blk_size;
    dim3 thr_size;
};

__global__
void calc_centroid_euclidean(float * d_samples, float * d_centroids, int * d_class, int samples, int features, int k){
    int sample_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(sample_idx >= samples) return;

    float * sample = &d_samples[sample_idx*features];
    float * centroid;

    int classification = 0;
    float result, best_result;

    for(int i=0;i<k;i++){
        centroid = &d_centroids[i*features]; 

        result = 0;
        for(int j=0;j<features;j++) result += pow(sample[j] - centroid[j], 2);
        result = sqrt(result);

        if(i == 0) best_result = result;
        else if(result < best_result){
            classification = i;
            best_result = result;
        }
    }

    d_class[sample_idx] = classification;
}

kernel_dim get_kernel_dimensions(int samples){
    //cudaDeviceProp properties;
    //cudaGetDeviceProperties(&properties,0);
    //properties.maxThreadsPerBlock;

    dim3 block_size(2,1,1);
    dim3 thread_size(5,1,1);

    struct kernel_dim dimension;
    dimension.blk_size = block_size;
    dimension.thr_size = thread_size;

    return dimension;
}

int* kmeans(float * h_samples, float * h_centroids, int * h_class, int samples, int features, int k){
    int samples_size = features * samples;
    int centroids_size = features * k;

    int * d_class;
    float * d_samples;
    float * d_centroids;

    cudaMalloc(&d_class, sizeof(int) * samples);
    cudaMalloc(&d_samples, sizeof(float) * samples_size);
    cudaMalloc(&d_centroids, sizeof(float) * centroids_size);

    cudaMemcpy(d_samples, h_samples, sizeof(float) * samples_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, sizeof(float) * centroids_size, cudaMemcpyHostToDevice);

    kernel_dim kernel = get_kernel_dimensions(samples);

    cout << endl << "Starting kmeans kernel...";
    calc_centroid_euclidean<<<kernel.blk_size, kernel.thr_size>>>(d_samples, d_centroids, d_class, samples, features, k);
    cout << "done" << endl << endl;
    
    cudaMemcpy(h_class, d_class, sizeof(int)*samples, cudaMemcpyDeviceToHost);

    cudaFree(d_class);
    cudaFree(d_centroids);
    cudaFree(d_samples);

    return h_class;
}
