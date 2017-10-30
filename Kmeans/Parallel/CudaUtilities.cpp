#include <cuda_runtime.h>

#include "CudaUtilities.h"

cudaDeviceProp get_device_properties(){
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,0);
    return device_prop;
}

kernel_dim get_vertical_kernel_dimensions(cudaDeviceProp device_prop, int samples){
    int max_threads = device_prop.maxThreadsPerBlock; 

    int blk_size = max_threads / samples;
    if (max_threads % samples != 0) blk_size++;

    dim3 block_size(blk_size,1,1);
    dim3 thread_size(max_threads,1,1);

    struct kernel_dim dimension;
    dimension.blk_size = block_size;
    dimension.thr_size = thread_size;

    return dimension;
}

kernel_dim get_horizontal_kernel_dimensions(cudaDeviceProp device_prop, int features, int k){
    int max_threads = device_prop.maxThreadsPerBlock; 

    int blk_size = max_threads / features*k;
    if (max_threads % features*k != 0) blk_size++;

    dim3 block_size(blk_size,1,1);
    dim3 thread_size(max_threads,1,1);

    struct kernel_dim dimension;
    dimension.blk_size = block_size;
    dimension.thr_size = thread_size;

    return dimension;
}
