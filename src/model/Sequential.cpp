#include "Sequential.h"

#include <random>

namespace mocr
{
    DTensor Sequential::feedForward(DTensor input)
    {
        auto &curr = input;

        for (auto it = m_Layers.cbegin(); it != m_Layers.cend(); ++it)
        {
            curr = (*it)->forward(curr);
        }

        return curr;
    }

    double Sequential::feedBackward(const DTensor &expected)
    {
        auto &output = m_Layers.back()->Output;

        if (m_ObjectiveFunc == LossFunc::MSE)
        {
            // Mean Square Error ...

            auto error = DTensor::sum(DTensor::map(DTensor::sub(expected, output), [](double x) { return x * x; })) * 0.5;
            auto deriv = DTensor::sub(output, expected);

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

        DTensor weights(1, connections, inputs);
        DTensor biases(1, connections, 1);

        for (int i = 0; i < weights.size(); i++)
            weights[i] = dist(mt) * scale;

        m_Layers.push_back(std::make_shared<FullyConLayer>(std::move(weights), std::move(biases)));
        m_Layers.push_back(std::make_shared<ActvLayer>(1, connections, 1, activation));
    }
}