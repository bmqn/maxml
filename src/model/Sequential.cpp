#include "Sequential.h"

#include "maths/Activations.h"

#include <random>

namespace mocr
{

    const Tensor<double> &Sequential::NeuronLayer::forward(const Tensor<double> &input)
    {
        Input = input;
        Output = mocr::add(mocr::matmul(Weights, input), Biases);

        return Output;
    }

    const Tensor<double> &Sequential::NeuronLayer::backward(const Tensor<double> &delta)
    {
        DWeights = mocr::matmul(delta, mocr::transpose(Input));
        DBiases = delta;
        Delta = mocr::matmul(mocr::transpose(Weights), delta);

        return Delta;
    }

    void Sequential::NeuronLayer::update(double learningRate)
    {
        Weights = mocr::sub(Weights, mocr::mult(DWeights, learningRate));
        Biases = mocr::sub(Biases, mocr::mult(DBiases, learningRate));
    }

    const Tensor<double> &Sequential::ActivationLayer::forward(const Tensor<double> &input)
    {
        Input = input;

        switch (Activation)
        {
        case ActivationFunc::SIGMOID:
            Output = mocr::map<double>(input, [](double x) { return sig(x); });
            break;
        case ActivationFunc::TANH:
            Output = mocr::map<double>(input, [](double x) { return tanh(x); });
            break;
        case ActivationFunc::RELU:
            Output = mocr::map<double>(input, [](double x) { return relu(x); });
            break;
        }

        return Output;
    }

    const Tensor<double> &Sequential::ActivationLayer::backward(const Tensor<double> &delta)
    {
        switch (Activation)
        {
        case ActivationFunc::SIGMOID:
            Delta = mocr::mult(mocr::map<double>(Output, [](double x) { return x * (1.0 - x); }), delta);
            break;
        case ActivationFunc::TANH:
            Delta = mocr::mult(mocr::map<double>(Input, [](double x) { return tanhPrime(x); }), delta);
            break;
        case ActivationFunc::RELU:
            Delta = mocr::mult(mocr::map<double>(Input, [](double x) { return reluPrime(x); }), delta);
            break;
        }

        return Delta;
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

        if (m_ObjectiveFunc == ObjectiveFunc::MSE)
        {
            // Mean Square Error ...

            auto error = mocr::sum(mocr::map<double>(mocr::sub(expected, output), [](double x) { return x * x; })) * 0.5;
            auto deriv = mocr::sub(output, expected);

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

    void Sequential::addLayer(int outputs, ActivationFunc activation)
    {
        int inputs = m_Layers.size() == 0 ? m_Inputs : m_Layers.back()->Size;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::normal_distribution dist(0.0, 1.0);

        double scale = std::sqrt(1.0 / inputs);

        Tensor<double> weights(1, outputs, inputs);
        Tensor<double> biases(1, outputs, 1);

        for (int i = 0; i < weights.Size; i++)
            weights[i] = dist(mt) * scale;

        m_Layers.push_back(std::make_shared<NeuronLayer>(outputs, std::move(weights), std::move(biases)));
        m_Layers.push_back(std::make_shared<ActivationLayer>(outputs, activation));
    }

}