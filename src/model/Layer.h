#ifndef H_LAYER_H
#define H_LAYER_H

#include "maths/Tensor.h"

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

    struct Layer
    {
        virtual const DTensor &forward(const DTensor &input) = 0;
        virtual const DTensor &backward(const DTensor &delta) = 0;
        virtual void update(double learningRate) = 0;

        DTensor Output;
        DTensor Input;

        DTensor DOutput;
        DTensor DInput;

        unsigned int Channels, Rows, Cols;
    };

    struct FullyConLayer : public Layer
    {
        FullyConLayer(DTensor &&weights, DTensor &&biases)
            : Weights(weights), Biases(biases)
        {
            Channels = weights.channels();
            Rows = weights.rows();
            Cols = weights.cols();
        }

        virtual const DTensor &forward(const DTensor &input) override;
        virtual const DTensor &backward(const DTensor &delta) override;
        virtual void update(double learningRate) override;

        DTensor Weights;
        DTensor Biases;
        DTensor DWeights;
        DTensor DBiases;
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

        virtual const DTensor &forward(const DTensor &input) override;
        virtual const DTensor &backward(const DTensor &delta) override;
        virtual void update(double learningRate) override{};

        ActivationFunc Activation;
    };
}

#endif