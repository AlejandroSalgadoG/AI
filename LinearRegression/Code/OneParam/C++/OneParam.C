#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <stdlib.h>

using namespace std;

float theta = 1;
float alpha = 0.1;

float x[3];
float y[3];

float ** getPoints(ifstream& file){
    string buffer;

    for(int i=0;i<3;i++){
        getline(file, buffer);
        istringstream reader(buffer);
        reader >> x[i] >> y[i];
    }
}

float h(float x){
    return theta * x;
}

float cost_function(){
    int num_points = 3;
    float summary = 0;

    for(int i=0;i<num_points;i++)
        summary += pow((h(x[i]) - y[i]), 2);

    return summary / (num_points * 2);
}

float cost_function_derivative(){
    int num_points = 3;
    float summary = 0;

    for(int i=0;i<num_points;i++)
        summary += (h(x[i]) - y[i]) * x[i];

    return summary / num_points;
}

void gradient_decend(){
    theta = theta - alpha * cost_function_derivative();
}

int main(int argc, char *argv[]){
    ifstream file(argv[1]);
    int iterations = atoi(argv[2]);
    
    getPoints(file);

    cout << "alpha = " << alpha << "\n" << endl;

    float err;

    for(int i=0;i<iterations;i++) {
        err = cost_function();
        cout << "theta = " << theta << " error = " << err << endl;
        gradient_decend();
    }

    return 0;
}
