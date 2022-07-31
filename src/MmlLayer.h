#pragma once

#include "maxml/MmlTensor.h"
#include "maxml/MmlSequential.h"

namespace maxml
{
	struct Layer
	{
		virtual ~Layer() {}

		virtual void forward(const Tensor &input, Tensor &output) = 0;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) = 0;

		virtual void update(float learningRate) = 0;
	};

	struct FullyConnectedLayer : public Layer
	{
		FullyConnectedLayer() = delete;
		FullyConnectedLayer(Tensor &&weights, Tensor &&biases);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override;

		Tensor DeltaWeights;
		Tensor DeltaBiases;

		Tensor Weights;
		Tensor Biases;
	};

	struct ConvolutionalLayer : public Layer
	{
		ConvolutionalLayer() = delete;
		ConvolutionalLayer(size_t inChannels, size_t outRows, size_t outCols, const Tensor &kernel);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override;

		size_t KernelChannels;
		size_t KernelRows;
		size_t KernelCols;

		Tensor KernelWindowed;
		Tensor InputWindowed;

		Tensor DeltaKernelWindowed;
		Tensor DeltaInputWindowed;
	};

	struct MaxPoolingLayer : public Layer
	{
		MaxPoolingLayer() = delete;
		MaxPoolingLayer(size_t tileWidth, size_t tileHeight);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override {};

		size_t TileWidth;
		size_t TileHeight;
	};

	struct FlattenLayer : public Layer
	{
		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override {};
	};

	struct ActivationLayer : public Layer
	{
		ActivationLayer() = delete;
		ActivationLayer(ActivationFunc activFunc);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override {};

		ActivationFunc ActivFunc;
	};
}