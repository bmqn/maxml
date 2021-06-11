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
            virtual const Tensor<double> &forward(const Tensor<double> &input) = 0;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) = 0;
            virtual void update(double learningRate) = 0;

            Tensor<double> Output;
            Tensor<double> Input;

            Tensor<double> DOutput;
            Tensor<double> DInput;

            unsigned int Channels, Rows, Cols;
        };

        struct FullyConLayer : public Layer
        {
            FullyConLayer(Tensor<double> &&weights, Tensor<double> &&biases)
                : Weights(weights), Biases(biases)
            {
                Channels = weights.channels();
                Rows = weights.rows();
                Cols = weights.cols();
            }

            virtual const Tensor<double> &forward(const Tensor<double> &input) override;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) override;
            virtual void update(double learningRate) override;

            Tensor<double> Weights;
            Tensor<double> Biases;
            Tensor<double> DWeights;
            Tensor<double> DBiases;
        };

        struct ConvLayer
        {
        };

        struct ActvLayer : public Layer
        {
            ActvLayer(unsigned int channels, unsigned int rows, unsigned int cols, ActivationFunc activation)
                : Activation(activation)
            {
                Channels = channels;
                Rows = rows;
                Cols = cols;
            }

            virtual const Tensor<double> &forward(const Tensor<double> &input) override;
            virtual const Tensor<double> &backward(const Tensor<double> &delta) override;
            virtual void update(double learningRate) override{};

            ActivationFunc Activation;
        };

    public:
        Sequential() = delete;
        Sequential(unsigned int inChannels, unsigned int inRows, unsigned int inCols, LossFunc objectiveFunc, double learningRate = 0.1)
            : m_InChannels(inChannels), m_InRows(inRows), m_InCols(inCols), m_ObjectiveFunc(objectiveFunc), m_LearningRate(learningRate)
        {
        }

        void addFullyConnectedLayer(int connections, ActivationFunc activation);

        Tensor<double> feedForward(const Tensor<double> &input);
        double feedBackward(const Tensor<double> &expected);

    private:
        std::vector<std::shared_ptr<Layer>> m_Layers;
        unsigned int m_InChannels, m_InRows, m_InCols;
        LossFunc m_ObjectiveFunc;
        double m_LearningRate;
    };
}

#endif