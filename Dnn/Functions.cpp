#include <math.h>
#include "Functions.h"

void Sigmoid::evaluate(double* x, double* ans, int size){
    for(int i=0;i<size;i++)
        x[i] = 1/(1+exp(x[i] * -1));
} 

double Sigmoid::derivative(double* x, int element){
    return x[element] * (1 - x[element]);
}

char const* Sigmoid::get_name(){
    return "Sigmoid";
}

void LessSquare::evaluate(double * y_hat, double* loss, int size){
    *loss = 0;
    for(int i=0;i<size;i++)
        *loss += pow(y[i] - y_hat[i], 2);
	*loss /= 2;
}

LessSquare::LessSquare(double * y){
	this->y = y;
}

double LessSquare::derivative(double * y_hat, int element){
	return y_hat[element] - y[element];;
}

char const* LessSquare::get_name(){
    return "LessSquare";
}

Function_creator::Function_creator(){
    activations_map["Sigmoid"] = &Function_creator::create_activation_function<Sigmoid>;

    loss_map["LessSquare"] = &Function_creator::create_loss_function<LessSquare>;
}

Activation* Function_creator::create_activation(std::string function_name){
    return (this->*activations_map[function_name])();
}

Loss* Function_creator::create_loss(std::string function_name, double* target){
    return (this->*loss_map[function_name])(target);
}

template<class T>
Activation* Function_creator::create_activation_function(){
    return new T();
}

template<class T>
Loss* Function_creator::create_loss_function(double* target){
    return new T(target);
}
