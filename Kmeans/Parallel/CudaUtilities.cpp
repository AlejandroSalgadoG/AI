#include <cuda_runtime.h>

#include "CudaUtilities.h"

cudaDeviceProp get_device_properties(){
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,0);
    return device_prop;
}

kernel_dim get_vertical_kernel_dimensions(cudaDeviceProp device_prop, int samples){
    //properties.maxThreadsPerBlock;

    dim3 block_size(2,1,1);
    dim3 thread_size(5,1,1);

    struct kernel_dim dimension;
    dimension.blk_size = block_size;
    dimension.thr_size = thread_size;

    return dimension;
}

kernel_dim get_horizontal_kernel_dimensions(cudaDeviceProp device_prop, int features, int k){

    dim3 block_size(1,1,1);
    dim3 thread_size(features*k,1,1);

    struct kernel_dim dimension;
    dimension.blk_size = block_size;
    dimension.thr_size = thread_size;

    return dimension;
}
