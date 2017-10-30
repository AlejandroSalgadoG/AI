#include <iostream>
#include <math.h>

using namespace std;

float euclidean_distance(float * sample, float * centroid, int features){
    float result = 0;
    for(int j=0;j<features;j++) result += pow(sample[j] - centroid[j], 2);
    return sqrt(result);
}

int* kmeans(float * h_samples, float * h_centroids, int * h_class, int samples, int features, int k){
    float *sample, *centroid;
    int classification;
    float result, best_result;
    float numerator, denominator;

    //Calculate distance to centroids
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

    //Calculate new centroids
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

    return h_class;
}
