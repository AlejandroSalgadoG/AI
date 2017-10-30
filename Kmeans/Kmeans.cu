__global__
void gpu_kmeans(float * d_samples, int features_size){
    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = j + i*features_size;

    d_samples[idx] += 1;
}

void kmeans(float * d_samples, int samples, int features){
    dim3 blockSize(samples,1,1);
    dim3 threadSize(features,1,1);

    gpu_kmeans<<<blockSize, threadSize>>>(d_samples, features);
}
