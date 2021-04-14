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
                    Delta = mocr::mult(mocr::map<double>(Input, [](double x) { return sigPrime(x); }), delta);
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
        Sequential(std::size_t inputs, double learningRate = 0.01)
            : m_Inputs(inputs), m_LearningRate(learningRate)
        {
            m_Layers.reserve(5);
        }

        void addLayer(int neurons, Activation activation);

        Tensor<double> feedForward(Tensor<double> &input);
        double feedBackward(const Tensor<double> &expected);

    private:
        std::vector<std::shared_ptr<Layer>> m_Layers;
        std::size_t m_Inputs;

        double m_LearningRate;
    };

    Tensor<double> Sequential::feedForward(Tensor<double> &input)
    {
        auto &inp = input;

        for (auto it = m_Layers.cbegin(); it != m_Layers.cend(); ++it)
        {
            inp = (*it)->forward(inp);
        }

        return inp;
    }

    double Sequential::feedBackward(const Tensor<double> &expected)
    {
        auto &output = m_Layers.back()->Output;

        auto diff = mocr::sub(output, expected);
        auto loss = mocr::sum(mocr::mult(mocr::map<double>(diff, [](double x) { return x * x; }), 0.5));

        auto &delta = diff;

        for (auto it = m_Layers.crbegin(); it != m_Layers.crend(); ++it)
        {
            delta = (*it)->backward(delta);

            (*it)->update(m_LearningRate);
        }

        return loss;
    }

    void Sequential::addLayer(int neurons, Activation activation)
    {
        int inputs = m_Layers.size() == 0 ? m_Inputs : m_Layers.back()->Size;

        std::default_random_engine generator;
        std::normal_distribution distribution(0.0, 1.0);

        Tensor<double> weights(1, neurons, inputs);
        Tensor<double> biases(1, neurons, 1);

        for (int i = 0; i < weights.Size; i++)
            weights[i] = distribution(generator);

        m_Layers.push_back(std::make_shared<NeuronLayer>(neurons, std::move(weights), std::move(biases)));
        m_Layers.push_back(std::make_shared<ActivationLayer>(neurons, activation));
    }

}

#endif