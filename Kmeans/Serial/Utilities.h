#pragma once

float * initialize_rand_samples(int seed, int range, int size);
float * initialize_rand_centroids(int seed, float * h_samples, int k, int samples, int features);
void print_samples(float * h_samples, int features, int samples);
void print_labeled_samples(float * h_samples, int * h_class, int features, int samples);
