#pragma once

#include "maxml/MmlTensor.h"
#include "maxml/MmlSequential.h"

#include "MmlLog.h"

#include <memory>
#include <cinttypes>

namespace maxml
{
	struct Layer
	{
		virtual ~Layer() {}

		virtual void forward(const FTensor &input, FTensor &output) = 0;
		virtual void backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta) = 0;

		virtual void update(float learningRate) = 0;
	};

	struct FullyConLayer : public Layer
	{
		FullyConLayer() = delete;
		FullyConLayer(FTensor &&weights, FTensor &&biases);

		virtual void forward(const FTensor &input, FTensor &output) override;
		virtual void backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta) override;

		virtual void update(float learningRate) override;

		FTensor DeltaWeights;
		FTensor DeltaBiases;

		FTensor Weights;
		FTensor Biases;
	};

	struct ConvLayer : public Layer
	{
		ConvLayer() = delete;
		ConvLayer(size_t inChannels, size_t outRows, size_t outCols, const FTensor &kernel);

		virtual void forward(const FTensor &input, FTensor &output) override;
		virtual void backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta) override;

		virtual void update(float learningRate) override;

		size_t KernelChannels;
		size_t KernelRows;
		size_t KernelCols;

		FTensor KernelWindowed;
		FTensor InputWindowed;

		FTensor DeltaKernelWindowed;
		FTensor DeltaInputWindowed;
	};

	struct MaxPoolLayer : public Layer
	{
		MaxPoolLayer() = delete;
		MaxPoolLayer(size_t tileWidth, size_t tileHeight);

		virtual void forward(const FTensor &input, FTensor &output) override;
		virtual void backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta) override;

		virtual void update(float learningRate) override{};

		size_t TileWidth;
		size_t TileHeight;
	};

	struct FlattenLayer : public Layer
	{
		virtual void forward(const FTensor &input, FTensor &output) override;
		virtual void backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta) override;

		virtual void update(float learningRate) override{};
	};

	struct ActvLayer : public Layer
	{
		ActvLayer() = delete;
		ActvLayer(ActivationFunc activation);

		virtual void forward(const FTensor &input, FTensor &output) override;
		virtual void backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta) override;

		virtual void update(float learningRate) override{};

		ActivationFunc Activation;
	};
}