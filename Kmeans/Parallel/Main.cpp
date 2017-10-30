#include "Utilities.h"
#include "Kmeans.h"

int main(int argc, char *argv[]){
    int samples, features, k, max_iterations;

    get_parameters(&samples, &features, &k, &max_iterations);

    float * h_samples = get_samples(samples, features);
    float * h_centroids = initialize_centroids(h_samples, k, features);
    int * h_class = new int[samples];

    //print_samples(h_samples, features, samples);
    h_class = kmeans(h_samples, h_centroids, h_class, samples, features, k, max_iterations);
    //print_labeled_samples(h_samples, h_class, features, samples);

    delete[] h_samples;
    delete[] h_centroids;
    delete[] h_class;

    return 0;
}
