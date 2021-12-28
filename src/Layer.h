#ifndef H_LAYER_H
#define H_LAYER_H

#include "maxml/Tensor.h"
#include "maxml/Sequential.h"

namespace maxml
{
	struct Layer
	{
		Layer() = delete;
		Layer(unsigned int channels , unsigned int rows, unsigned int cols)
			: Channels(channels), Rows(rows), Cols(cols)
		{
		}

		virtual ~Layer() {}

		virtual void forward(const DTensor& input, DTensor& output) = 0;
		virtual void backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta) = 0;

		virtual void update(double learningRate) = 0;

		unsigned int Channels, Rows, Cols;
	};

	struct FullyConLayer : public Layer
	{
		FullyConLayer() = delete;
		FullyConLayer(DTensor&& weights, DTensor&& biases)
			: Layer(weights.channels(), weights.rows(), 1)
			, DeltaWeights(weights.channels(), weights.rows(), weights.cols())
			, DeltaBiases(weights.channels(), weights.rows(), 1)
			, Weights(std::forward<DTensor>(weights))
			, Biases(std::forward<DTensor>(biases))
		{
		}

		virtual void forward(const DTensor& input, DTensor& output) override;
		virtual void backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta) override;

		virtual void update(double learningRate) override;

		DTensor DeltaWeights;
		DTensor DeltaBiases;

		DTensor Weights;
		DTensor Biases;
	};

	struct ConvLayer
	{
	};

	struct ActvLayer : public Layer
	{
		ActvLayer() = delete;
		ActvLayer(unsigned int channels, unsigned int rows, unsigned int cols, ActivationFunc activation)
			: Layer(channels, rows, cols), Activation(activation)
		{
		}

		virtual void forward(const DTensor& input, DTensor& output) override;
		virtual void backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta) override;

		virtual void update(double learningRate) override {};

		ActivationFunc Activation;
	};
}

#endif