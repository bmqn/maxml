#pragma once

#include "layers/Layer.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/ReluLayer.h"
#include "utils/Tensor.h"

#include <vector>
#include <functional>
#include <utility>
#include <map>

class Network
{
public:
	class Builder;

	Network(std::vector<Tensor<float>>&& data, std::vector<Tensor<float>>&& gradient, std::vector<std::shared_ptr<Layer>>&& layers)
		: data_(std::move(data)), gradient_(std::move(gradient)), layers_(std::move(layers)), expected_(nullptr)
	{
	}

	void setDataCallbacks(std::function<void(Tensor<float>&)> inputCallback, std::function<void(Tensor<float>&)> expecCallback)
	{
		this->inputCallback_ = inputCallback;
		this->expecCallback_ = expecCallback;
	}

	void train();
	Tensor<float> predict(const Tensor<float>& input);

public:
	static Builder make(int channels, int width, int height, float learningRate = 0.01f);

private:
	void forwardPropagate();
	void backwardPropagate();
	void updateParameters();

private:
	std::vector<std::shared_ptr<Layer>> layers_;	// Stores the layers with the input at index zero.
	std::vector<Tensor<float>>			data_;		// Stores the input and output of each layer.
	std::vector<Tensor<float>>			gradient_;	// Stores the input and output gradient of each layer.

	std::shared_ptr<Tensor<float>>		expected_;	// Stores the expected output for the CURRENT input.

	std::function<void(Tensor<float>&)> inputCallback_;
	std::function<void(Tensor<float>&)> expecCallback_;
};

class Network::Builder
{
private:
	struct InpLayer
	{
		int inputChannels_;
		int inputWidth_;
		int inputHeight_;
	};

	struct ConvLayer
	{
		int kernelSize_;
		int kernelNum_;
	};

	struct FullConLayer
	{
		int size_;
	};

public:

	Builder(int channels, int width, int height, float learningRate)
		: inpLayer_{channels, width, height}, learningRate_(learningRate)
	{
	}

	Builder& addConvLayer(int kernelSize, int kernelNum)
	{
		convLayers_.push_back({ kernelSize, kernelNum });

		int index			= convLayers_.size() - 1;
		std::string name	= "conv";
		
		layerInfo_[layerCount_++] = { name, index };

		return *this;
	}

	Builder& addMaxPoolLayer(int stride);

	Builder& addReluLayer()
	{
		std::string name	= "relu";

		layerInfo_[layerCount_++] = { name, {} };

		return *this;
	}

	Builder& addFullyConnectedLayer(int size)
	{
		fcLayers_.push_back({ size });

		int index			= fcLayers_.size() - 1;
		std::string name	= "fc";

		layerInfo_[layerCount_++] = { name, index };

		return *this;
	}

	Builder& addSoftmaxLayer();

	Network build()
	{
		// Build the network...

		std::vector<Tensor<float>>			data;
		std::vector<Tensor<float>>			gradient;
		std::vector<std::shared_ptr<Layer>>	layers;

		// Input layer...
		data.push_back(Tensor<float>(inpLayer_.inputChannels_, inpLayer_.inputWidth_, inpLayer_.inputHeight_));
		gradient.push_back(Tensor<float>(Tensor<float>(inpLayer_.inputChannels_, inpLayer_.inputWidth_, inpLayer_.inputHeight_)));

		int inputChannels = inpLayer_.inputChannels_;
		int inputWidth = inpLayer_.inputWidth_;
		int inputHeight = inpLayer_.inputHeight_;

		std::cout
			<< "0: input, "
			<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << "), "
			<< std::endl;

		int outputChannels;
		int outputWidth;
		int outputHeight;

		for (int i = 0; i < layerInfo_.size(); i++)
		{
			auto layer = layerInfo_[i];

			std::string name = layer.first;
			int index = layer.second;

			if (name == "conv")
			{
				ConvLayer cv = convLayers_[index];

				outputChannels = cv.kernelNum_;
				outputWidth = inputWidth / cv.kernelSize_;
				outputHeight = inputHeight / cv.kernelSize_;

				std::cout
					<< i + 1 << ": conv, "
					<< "kernel size: " << cv.kernelSize_ << ", "
					<< "kernel num: " << cv.kernelNum_ << ", "
					<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << ") -> "
					<< "(" << outputChannels << ", " << outputWidth << ", " << outputHeight << ")"
					<< std::endl;

				// Push back the output tensor for this layer...
				data.push_back(Tensor<float>(Tensor<float>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<float>(Tensor<float>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<ConvolutionLayer>(cv.kernelSize_, cv.kernelNum_));
			}
			else if (name == "fc")
			{
				FullConLayer fc = fcLayers_[index];

				outputChannels = fc.size_;
				outputWidth = 1;
				outputHeight = 1;

				int inputSize = inputChannels * inputWidth * inputHeight;

				std::cout
					<< i + 1 << ": fc, "
					<< "size: " << fc.size_ << ", "
					<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << ") -> "
					<< "(" << outputChannels << ", " << outputWidth << ", " << outputHeight << ")"
					<< std::endl;

				// Push back the output tensor for this layer...
				data.push_back(Tensor<float>(Tensor<float>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<float>(Tensor<float>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<FullyConnectedLayer>(inputSize, fc.size_));
			}
			else if (name == "relu")
			{
				outputChannels = inputChannels;
				outputWidth = inputWidth;
				outputHeight = inputHeight;

				std::cout
					<< i + 1 << ": relu, "
					<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << ") -> "
					<< "(" << outputChannels << ", " << outputWidth << ", " << outputHeight << ")"
					<< std::endl;

				// Push back the output tensor for this layer...
				data.push_back(Tensor<float>(Tensor<float>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<float>(Tensor<float>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<ReluLayer>());
			}

			inputChannels = outputChannels;
			inputWidth = outputWidth;
			inputHeight = outputHeight;
		}

		return Network(std::move(data), std::move(gradient), std::move(layers));
	}


private:

	float learningRate_;
	InpLayer inpLayer_;

	std::vector<ConvLayer>		convLayers_;
	std::vector<FullConLayer>	fcLayers_;

	int layerCount_{ 0 };
	std::map<int, std::pair<std::string, int>> layerInfo_;

};