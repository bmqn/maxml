#pragma once

#include "maxml/MmlTensor.h"
#include "maxml/MmlSequential.h"

namespace maxml
{
	struct Layer
	{
		Layer() = delete;
		Layer(size_t channels , size_t rows, size_t cols)
			: Channels(channels), Rows(rows), Cols(cols)
		{
		}

		virtual ~Layer() {}

		virtual void forward(const DTensor& input, DTensor& output) = 0;
		virtual void backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta) = 0;

		virtual void update(double learningRate) = 0;

		size_t Channels, Rows, Cols;
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
		ActvLayer(size_t channels, size_t rows, size_t cols, ActivationFunc activation)
			: Layer(channels, rows, cols), Activation(activation)
		{
		}

		virtual void forward(const DTensor& input, DTensor& output) override;
		virtual void backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta) override;

		virtual void update(double learningRate) override {};

		ActivationFunc Activation;
	};
}