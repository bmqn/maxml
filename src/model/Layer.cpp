#include "Layer.h"

#include "maths/Activations.h"

namespace mocr
{
    const DTensor &FullyConLayer::forward(const DTensor &input)
    {
        Input = input;
        Output = DTensor::add(DTensor::matmul(Weights, input), Biases);

        return Output;
    }

    const DTensor &FullyConLayer::backward(const DTensor &delta)
    {
        DWeights = DTensor::matmul(delta, DTensor::transpose(Input));
        DBiases = delta;
        DOutput = delta;
        DInput = DTensor::matmul(DTensor::transpose(Weights), delta);

        return DInput;
    }

    void FullyConLayer::update(double learningRate)
    {
        Weights = DTensor::sub(Weights, DTensor::mult(DWeights, learningRate));
        Biases = DTensor::sub(Biases, DTensor::mult(DBiases, learningRate));
    }

    const DTensor &ActvLayer::forward(const DTensor &input)
    {
        Input = input;

        switch (Activation)
        {
        case ActivationFunc::SIGMOID:
            Output = DTensor::map(input, [](double x) { return sig(x); });
            break;
        case ActivationFunc::TANH:
            Output = DTensor::map(input, [](double x) { return tanh(x); });
            break;
        case ActivationFunc::RELU:
            Output = DTensor::map(input, [](double x) { return relu(x); });
            break;
        }

        return Output;
    }

    const DTensor &ActvLayer::backward(const DTensor &delta)
    {
        DOutput = delta;

        switch (Activation)
        {
        case ActivationFunc::SIGMOID:
            DInput = DTensor::mult(DTensor::map(Output, [](double x) { return x * (1.0 - x); }), delta);
            break;
        case ActivationFunc::TANH:
            DInput = DTensor::mult(DTensor::map(Input, [](double x) { return tanhPrime(x); }), delta);
            break;
        case ActivationFunc::RELU:
            DInput = DTensor::mult(DTensor::map(Input, [](double x) { return reluPrime(x); }), delta);
            break;
        }

        return DInput;
    }
}