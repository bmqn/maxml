#ifndef H_SEQUENTIAL_H
#define H_SEQUENTIAL_H

#include "layers/Layer.h"

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
            virtual Tensor<double> forward(const Tensor<double> &input) = 0;
            virtual Tensor<double> backward(const Tensor<double> &delta) = 0;
            virtual void update(double learningRate) = 0;

            Layer() = delete;
            Layer(std::size_t size) : Size(size) {}

            std::size_t Size;
            Tensor<double> Input;
            Tensor<double> Output;
        };

        struct NeuronLayer : public Layer
        {
            NeuronLayer(std::size_t size, Tensor<double> &&weights, Tensor<double> &&biases)
                : Layer(size), Weights(weights), Biases(biases)
            {
            }

            virtual Tensor<double> forward(const Tensor<double> &input) override
            {
                Input = input;
                Output = mocr::add(mocr::matmul(Weights, input), Biases);

                return Output;
            }

            virtual Tensor<double> backward(const Tensor<double> &delta) override
            {
                dWeights = mocr::matmul(delta, mocr::transpose(Input));
                dBiases = delta;

                return mocr::matmul(mocr::transpose(Weights), delta);
            }

            virtual void update(double learningRate) override
            {
                Weights = mocr::sub(Weights, mocr::mult(dWeights, learningRate));
                Biases = mocr::sub(Biases, mocr::mult(dBiases, learningRate));
            }

            Tensor<double> Weights;
            Tensor<double> Biases;

            Tensor<double> dWeights;
            Tensor<double> dBiases;
        };

        struct ActivationLayer : public Layer
        {
            ActivationLayer(std::size_t size, Activation activation)
                : Layer(size), Acti(activation)
            {
            }

            virtual Tensor<double> forward(const Tensor<double> &input) override
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

            virtual Tensor<double> backward(const Tensor<double> &delta) override
            {
                switch (Acti)
                {
                case Activation::SIGMOID:
                    return mocr::mult(mocr::map<double>(Input, [](double x) { return sigPrime(x); }), delta);
                    break;
                case Activation::TANH:
                    return mocr::mult(mocr::map<double>(Input, [](double x) { return tanhPrime(x); }), delta);
                    break;
                case Activation::RELU:
                    return mocr::mult(mocr::map<double>(Input, [](double x) { return reluPrime(x); }), delta);
                    break;
                }
            }

            virtual void update(double learningRate) override{};

            Activation Acti;

        private:
            static double sig(double x)
            {
                return 1.0 / (1.0 + std::exp(-x));
            }

            static double sigPrime(double x)
            {
                return sig(x) * (1.0 - sig(x));
            }

            static double relu(double x)
            {
                return x < 0.0 ? 0.0 : x;
            }

            static double reluPrime(double x)
            {
                return x < 0.0 ? 0.0 : 1.0;
            }

            static double tanh(double x)
            {
                return std::tanh(x);
            }

            static double tanhPrime(double x)
            {
                return 1.0 / (std::cosh(x) * std::cosh(x));
            }
        };

    public:
        Sequential(std::size_t inputs) : m_Inputs(inputs)
        {
            m_Layers.reserve(5);
        }

        void addLayer(int neurons, Activation activation);

        Tensor<double> feedForward(const Tensor<double> &input);
        double feedBackward(const Tensor<double> &expected);

    private:
        std::vector<std::shared_ptr<Layer>> m_Layers;
        std::size_t m_Inputs;
    };

    Tensor<double> Sequential::feedForward(const Tensor<double> &input)
    {
        Tensor<double> inp = input;

        for (auto it = m_Layers.begin(); it != m_Layers.end(); ++it)
        {
            inp = (*it)->forward(inp);
        }

        return inp;
    }

    double Sequential::feedBackward(const Tensor<double> &expected)
    {
        Tensor<double> output = m_Layers.back()->Output;
        Tensor<double> delta = mocr::sub(output, expected);

        double loss = mocr::sum(mocr::mult(mocr::map<double>(mocr::sub(output, expected), [](double x) { return x * x; }), 0.5));

        for (auto it = m_Layers.rbegin(); it != m_Layers.rend(); ++it)
        {
            delta = (*it)->backward(delta);

            (*it)->update(0.01);
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