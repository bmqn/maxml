#pragma once

#include "maxml/MmlTensor.h"
#include "maxml/MmlSequential.h"

#include "MmlLog.h"

#include <cstdint>
#include <memory>

namespace maxml
{
	struct Layer
	{
		virtual ~Layer() {}

		virtual void forward(const Tensor &input, Tensor &output) = 0;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) = 0;

		virtual void update(float learningRate) = 0;
	};

	struct FullyConLayer : public Layer
	{
		FullyConLayer() = delete;
		FullyConLayer(Tensor &&weights, Tensor &&biases);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override;

		Tensor DeltaWeights;
		Tensor DeltaBiases;

		Tensor Weights;
		Tensor Biases;
	};

	struct ConvLayer : public Layer
	{
		ConvLayer() = delete;
		ConvLayer(size_t inChannels, size_t outRows, size_t outCols, const Tensor &kernel);

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

	struct MaxPoolLayer : public Layer
	{
		MaxPoolLayer() = delete;
		MaxPoolLayer(size_t tileWidth, size_t tileHeight);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override{};

		size_t TileWidth;
		size_t TileHeight;
	};

	struct FlattenLayer : public Layer
	{
		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override{};
	};

	struct ActvLayer : public Layer
	{
		ActvLayer() = delete;
		ActvLayer(ActivationFunc activation);

		virtual void forward(const Tensor &input, Tensor &output) override;
		virtual void backward(const Tensor &input, const Tensor &output, Tensor &inputDelta, const Tensor &outputDelta) override;

		virtual void update(float learningRate) override{};

		ActivationFunc Activation;
	};
}