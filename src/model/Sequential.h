#ifndef H_SEQUENTIAL_H
#define H_SEQUENTIAL_H

#include "maths/Tensor.h"

#include <vector>
#include <memory>

namespace mocr
{
    enum class ActivationFunc
    {
        SIGMOID,
        TANH,
        RELU
    };

    enum class LossFunc
    {
        MSE
    };

    class Sequential
    {
    private:
        struct Layer
        {
            Layer() = delete;
            Layer(std::size_t size) : Size(size) {}

            virtual const Tensor<double> &forward(const Tensor<double> &input) = 0;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) = 0;
            virtual void update(double learningRate) = 0;

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

            virtual const Tensor<double> &forward(const Tensor<double> &input) override;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) override;
            virtual void update(double learningRate) override;

            Tensor<double> Weights;
            Tensor<double> Biases;
            Tensor<double> DWeights;
            Tensor<double> DBiases;
        };

        struct ActivationLayer : public Layer
        {
            ActivationLayer(std::size_t size, ActivationFunc activation)
                : Layer(size), Activation(activation)
            {
            }

            virtual const Tensor<double> &forward(const Tensor<double> &input) override;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) override;
            virtual void update(double learningRate) override{};

            ActivationFunc Activation;
        };

    public:
        Sequential() = delete;
        Sequential(std::size_t inputs, LossFunc objectiveFunc, double learningRate = 0.1)
            : m_Inputs(inputs), m_ObjectiveFunc(objectiveFunc), m_LearningRate(learningRate)
        {
        }

        void addLayer(int outputs, ActivationFunc activation);
        Tensor<double> feedForward(const Tensor<double> &input);
        double feedBackward(const Tensor<double> &expected);

    private:
        std::vector<std::shared_ptr<Layer>> m_Layers;
        std::size_t m_Inputs;
        LossFunc m_ObjectiveFunc;
        double m_LearningRate;
    };
}

#endif