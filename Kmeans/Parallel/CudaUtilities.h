#pragma once

struct kernel_dim{
    dim3 blk_size;
    dim3 thr_size;
};

cudaDeviceProp get_device_properties();
kernel_dim get_calculation_kernel_dimensions(cudaDeviceProp device_prop, int samples);
kernel_dim get_movement_kernel_dimensions(cudaDeviceProp device_prop, int features, int k);
