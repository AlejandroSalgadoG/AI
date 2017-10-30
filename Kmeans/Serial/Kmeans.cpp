#include <iostream>
#include <math.h>
#include <cstring>

using namespace std;

float euclidean_distance(float * sample, float * centroid, int features){
    float result = 0;
    for(int j=0;j<features;j++) result += pow(sample[j] - centroid[j], 2);
    return sqrt(result);
}

void calculate_distance(float * h_samples, float * h_centroids, int * h_class, int samples, int features, int k){
    float *sample, *centroid;
    int classification;
    float result, best_result;

    for(int sample_idx=0;sample_idx<samples;sample_idx++){
        sample = &h_samples[sample_idx*features];

        classification = 0;
        for(int i=0;i<k;i++){
            centroid = &h_centroids[i*features];
            result = euclidean_distance(sample, centroid, features);

            if(i == 0) best_result = result;
            else if(result < best_result){
                classification = i;
                best_result = result;
            }
        }

        h_class[sample_idx] = classification;
    }
}

void compare_classes(int * h_class, int * h_past_class, int samples, bool * are_equal){
    bool equal_status = true;

    for(int i=0;i<samples;i++){
        if(h_class[i] != h_past_class[i]){
            equal_status = false;
            break;
        }
    }

    *are_equal = equal_status;
}

void move_centroids(float * h_samples, float * h_centroids, int * h_class, int samples, int features, int k){
    float numerator, denominator;

    for(int feat=0;feat<features;feat++){
        for(int cent=0;cent<k;cent++){
            numerator = 0;
            denominator = 0;

            for(int i=0;i<samples;i++){
                if(h_class[i] == cent){
                    numerator += h_samples[feat + i*features];
                    denominator += 1;
                }
            }

            h_centroids[feat + cent*features] = numerator/denominator;
        }
    }
}

int* kmeans(float * h_samples, float * h_centroids, int * h_class, int samples, int features, int k, int max_iterations){
    int * h_past_class = new int[samples];
    memcpy(h_past_class, h_class, sizeof(int)*samples);

    bool are_equal = false;
    int iteration = 0;

    while(!are_equal && iteration++ < max_iterations){
        cout << "iteration " << iteration << endl;

        calculate_distance(h_samples, h_centroids, h_class, samples, features, k);

        compare_classes(h_class, h_past_class, samples, &are_equal);

        if(!are_equal){
            memcpy(h_past_class, h_class, sizeof(float)*samples);

            move_centroids(h_samples, h_centroids, h_class, samples, features, k);
        }
    }

    return h_class;
}
