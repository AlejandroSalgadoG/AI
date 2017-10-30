#include "Utilities.h"
#include "Kmeans.h"

int main(int argc, char *argv[]){
    int features = 2;
    int samples = 19;

    int k = 2;
    int seed = 1;
    int rand_range = 10;

    float * h_samples = initialize_rand_samples(seed, rand_range, features*samples);
    float * h_centroids = initialize_rand_centroids(seed, h_samples, k, samples, features);
    int * h_class = new int[samples];

    print_samples(h_samples, features, samples);
    h_class = kmeans(h_samples, h_centroids, h_class, samples, features, k);
    print_labeled_samples(h_samples, h_class, features, samples);

    return 0;
}
