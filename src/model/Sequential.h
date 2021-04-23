#ifndef H_SEQUENTIAL_H
#define H_SEQUENTIAL_H

#include "layers/Layer.h"

#include "maths/Activations.h"

#include <vector>
#include <random>
#include <iostream>
#include <memory>

namespace mocr
{

    enum class Activation
    {
        SIGMOID,
        TANH,
        RELU
    };

    class Sequential
    {
    private:
        struct Layer
        {
            virtual const Tensor<double> &forward(const Tensor<double> &input) = 0;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) = 0;
            virtual void update(double learningRate) = 0;

            Layer() = delete;
            Layer(std::size_t size) : Size(size) {}

            std::size_t Size;
            Tensor<double> Input;
            Tensor<double> Output;
            Tensor<double> Delta;
        };

        struct NeuronLayer : public Layer
        {
            NeuronLayer(std::size_t size, Tensor<double> &&weights, Tensor<double> &&biases)
                : Layer(size), Weights(weights), Biases(biases)
            {
            }

            virtual const Tensor<double> &forward(const Tensor<double> &input) override
            {
                Input = input;
                Output = mocr::add(mocr::matmul(Weights, input), Biases);

                return Output;
            }

            virtual const Tensor<double> &backward(const Tensor<double> &delta) override
            {
                DWeights = mocr::matmul(delta, mocr::transpose(Input));
                DBiases = delta;
                Delta = mocr::matmul(mocr::transpose(Weights), delta);

                return Delta;
            }

            virtual void update(double learningRate) override
            {
                Weights = mocr::sub(Weights, mocr::mult(DWeights, learningRate));
                Biases = mocr::sub(Biases, mocr::mult(DBiases, learningRate));
            }

            Tensor<double> Weights;
            Tensor<double> Biases;
            Tensor<double> DWeights;
            Tensor<double> DBiases;
        };

        struct ActivationLayer : public Layer
        {
            ActivationLayer(std::size_t size, Activation activation)
                : Layer(size), Acti(activation)
            {
            }

            virtual const Tensor<double> &forward(const Tensor<double> &input) override
            {
                Input = input;

                switch (Acti)
                {
                case Activation::SIGMOID:
                    Output = mocr::map<double>(input, [](double x) { return sig(x); });
                    break;
                case Activation::TANH:
                    Output = mocr::map<double>(input, [](double x) { return tanh(x); });
                    break;
                case Activation::RELU:
                    Output = mocr::map<double>(input, [](double x) { return relu(x); });
                    break;
                }

                return Output;
            }

            virtual const Tensor<double> &backward(const Tensor<double> &delta) override
            {
                switch (Acti)
                {
                case Activation::SIGMOID:
                    Delta = mocr::mult(mocr::map<double>(Output, [](double x) { return x * (1.0 - x); }), delta);
                    break;
                case Activation::TANH:
                    Delta = mocr::mult(mocr::map<double>(Input, [](double x) { return tanhPrime(x); }), delta);
                    break;
                case Activation::RELU:
                    Delta = mocr::mult(mocr::map<double>(Input, [](double x) { return reluPrime(x); }), delta);
                    break;
                }

                return Delta;
            }

            virtual void update(double learningRate) override{};

            Activation Acti;
        };

    public:
        Sequential() = delete;
        Sequential(std::size_t inputs, double learningRate = 0.1)
            : m_Inputs(inputs), m_LearningRate(learningRate)
        {
            m_Layers.reserve(5);
        }

        void addLayer(int outputs, Activation activation);

        Tensor<double> feedForward(const Tensor<double> &input);
        double feedBackward(const Tensor<double> &expected);

    private:
        std::vector<std::shared_ptr<Layer>> m_Layers;
        std::size_t m_Inputs;

        double m_LearningRate;
    };

    Tensor<double> Sequential::feedForward(const Tensor<double> &input)
    {
        // Make a copy ... TODO: this is inefficient.
        auto inp = Tensor(input);

        auto &curr = inp;

        for (auto it = m_Layers.cbegin(); it != m_Layers.cend(); ++it)
        {
            curr = (*it)->forward(curr);
        }

        // Softmax ...
        // auto sum = mocr::sum(mocr::map<double>(curr, [](double x) { return std::exp(x); }));
        // auto out = mocr::map<double>(curr, [=](double x) { return std::exp(x) / sum; });

        return curr;
    }

    double Sequential::feedBackward(const Tensor<double> &expected)
    {
        auto &output = m_Layers.back()->Output;

        // Mean Square Error ...
        auto error = mocr::sum(mocr::map<double>(mocr::sub(expected, output), [](double x) { return x * x; })) * 0.5;
        auto deriv = mocr::sub(output, expected);

        // Softmax + Cross Entropy ...
        // auto sum = mocr::sum(mocr::map<double>(output, [](double x) { return std::exp(x); }));
        // auto out = mocr::map<double>(output, [=](double x) { return std::exp(x) / sum; });

        // auto error = -1.0 * mocr::sum(mocr::mult(expected, mocr::map<double>(out, [](double x) { return std::log(x); })));
        // auto deriv = mocr::sub(out, expected);

        auto &delta = deriv;

        for (auto it = m_Layers.crbegin(); it != m_Layers.crend(); ++it)
        {
            delta = (*it)->backward(delta);

            (*it)->update(m_LearningRate);
        }

        return error;
    }

    void Sequential::addLayer(int outputs, Activation activation)
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

#endif