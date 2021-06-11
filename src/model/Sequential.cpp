#include "Sequential.h"

#include "maths/Activations.h"

#include <random>

namespace mocr
{

    const Tensor<double> &Sequential::FullyConLayer::forward(const Tensor<double> &input)
    {
        Input = input;
        Output = Tensor<double>::add(Tensor<double>::matmul(Weights, input), Biases);

        return Output;
    }

    const Tensor<double> &Sequential::FullyConLayer::backward(const Tensor<double> &delta)
    {

        DWeights = Tensor<double>::matmul(delta, Tensor<double>::transpose(Input));
        DBiases = delta;
        DOutput = delta;
        DInput = Tensor<double>::matmul(Tensor<double>::transpose(Weights), delta);

        return DInput;
    }

    void Sequential::FullyConLayer::update(double learningRate)
    {
        Weights = Tensor<double>::sub(Weights, Tensor<double>::mult(DWeights, learningRate));
        Biases = Tensor<double>::sub(Biases, Tensor<double>::mult(DBiases, learningRate));
    }

    const Tensor<double> &Sequential::ActvLayer::forward(const Tensor<double> &input)
    {
        Input = input;

        switch (Activation)
        {
        case ActivationFunc::SIGMOID:
            Output = Tensor<double>::map(input, [](double x) { return sig(x); });
            break;
        case ActivationFunc::TANH:
            Output = Tensor<double>::map(input, [](double x) { return tanh(x); });
            break;
        case ActivationFunc::RELU:
            Output = Tensor<double>::map(input, [](double x) { return relu(x); });
            break;
        }

        return Output;
    }

    const Tensor<double> &Sequential::ActvLayer::backward(const Tensor<double> &delta)
    {
        DOutput = delta;

        switch (Activation)
        {
        case ActivationFunc::SIGMOID:
            DInput = Tensor<double>::mult(Tensor<double>::map(Output, [](double x) { return x * (1.0 - x); }), delta);
            break;
        case ActivationFunc::TANH:
            DInput = Tensor<double>::mult(Tensor<double>::map(Input, [](double x) { return tanhPrime(x); }), delta);
            break;
        case ActivationFunc::RELU:
            DInput = Tensor<double>::mult(Tensor<double>::map(Input, [](double x) { return reluPrime(x); }), delta);
            break;
        }

        return DInput;
    }

    Tensor<double> Sequential::feedForward(const Tensor<double> &input)
    {
        // Make a copy ... TODO: this is inefficient.
        auto inp = Tensor(input);

        auto &curr = inp;

        for (auto it = m_Layers.cbegin(); it != m_Layers.cend(); ++it)
        {
            curr = (*it)->forward(curr);
        }

        return curr;
    }

    double Sequential::feedBackward(const Tensor<double> &expected)
    {
        auto &output = m_Layers.back()->Output;

        if (m_ObjectiveFunc == LossFunc::MSE)
        {
            // Mean Square Error ...

            auto error = Tensor<double>::sum(Tensor<double>::map(Tensor<double>::sub(expected, output), [](double x) { return x * x; })) * 0.5;
            auto deriv = Tensor<double>::sub(output, expected);

            auto &delta = deriv;

            for (auto it = m_Layers.crbegin(); it != m_Layers.crend(); ++it)
            {
                delta = (*it)->backward(delta);

                (*it)->update(m_LearningRate);
            }

            return error;
        }
        else
        {
            return INFINITY;
        }
    }

    void Sequential::addFullyConnectedLayer(int connections, ActivationFunc activation)
    {
        // TODO: conversion layer if previous is conv layer.

        auto inputs = m_Layers.size() == 0 ? m_InRows : m_Layers.back()->Rows;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::normal_distribution dist(0.0, 1.0);

        double scale = std::sqrt(1.0 / inputs);

        Tensor<double> weights(1, connections, inputs);
        Tensor<double> biases(1, connections, 1);

        for (int i = 0; i < weights.size(); i++)
            weights[i] = dist(mt) * scale;

        m_Layers.push_back(std::make_shared<FullyConLayer>(std::move(weights), std::move(biases)));
        m_Layers.push_back(std::make_shared<ActvLayer>(1, connections, 1, activation));
    }

}