#pragma once

void get_parameters(int * samples, int * features, int * k, int * max_iterations);
float * get_samples(int samples, int features);
float * initialize_centroids(float * h_samples, int k, int features);
void print_samples(float * h_samples, int features, int samples);
void print_labeled_samples(float * h_samples, int * h_class, int features, int samples);
