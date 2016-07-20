#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <stdlib.h>

using namespace std;

float theta_0 = 1;
float theta_1 = 1;
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
    return theta_0 + theta_1 * x;
}

float cost_function(){
    int num_points = 3;
    float summary = 0;

    for(int i=0;i<num_points;i++)
        summary += pow((h(x[i]) - y[i]), 2);

    return summary / (num_points * 2);
}

float cost_function_derivative_theta0(){
    int num_points = 3;
    float summary = 0;

    for(int i=0;i<num_points;i++)
        summary += (h(x[i]) - y[i]) * x[i];

    return summary / num_points;
}

float cost_function_derivative_theta1(){
    int num_points = 3;
    float summary = 0;

    for(int i=0;i<num_points;i++)
        summary += (h(x[i]) - y[i]);

    return summary / num_points;
}

void gradient_decend(){
    float temp_0 = theta_0 - alpha * cost_function_derivative_theta0();
    float temp_1 = theta_1 - alpha * cost_function_derivative_theta1();

    theta_0 = temp_0;
    theta_1 = temp_1;
}

int main(int argc, char *argv[]){
    ifstream file(argv[1]);
    int iterations = atoi(argv[2]);

    getPoints(file);

    cout << "alpha = " << alpha << "\n" << endl;

    float err;

    for(int i=0;i<iterations;i++) {
        err = cost_function();
        cout << "theta_0 = " << theta_0 << endl;
        cout << "theta_1 = " << theta_1 << endl;
        cout << "error = " << err << endl;
        cout << endl;
        gradient_decend();
    }

    return 0;

}
