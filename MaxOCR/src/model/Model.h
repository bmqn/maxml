#pragma once

#include "../layers/Layer.h"
#include "../layers/ConvolutionLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include "../layers/ReluLayer.h"

#include "../utils/Tensor.h"

#include <functional>
#include <map>

template <typename T>
class Model
{
public:
	class Builder;

	Model(std::vector<Tensor<T>>&& data, std::vector<Tensor<T>>&& gradient, std::vector<std::shared_ptr<Layer<T>>>&& layers,
		int outputChannels, int outputWidth, int outputHeight, T learningRate)
		: data_(std::move(data)), gradient_(std::move(gradient)), layers_(std::move(layers)),
		expected_(outputChannels, outputWidth, outputHeight), learningRate_(learningRate)
	{
	}

	void setDataCallbacks(std::function<void(Tensor<T>&)> inputCallback, std::function<void(Tensor<T>&)> expecCallback)
	{
		this->inputCallback_ = inputCallback;
		this->expecCallback_ = expecCallback;
	}

	void train()
	{
		// TODO: Add my own assert!
		assert(inputCallback_ && expecCallback_ && !data_.empty() && !layers_.empty());

		inputCallback_(data_[0]);
		expecCallback_(expected_);

		forwardPropagate();
		backwardPropagate();

		updateParameters();
	}

	T predict(const Tensor<T>& input, const Tensor<T>& expected)
	{
		assert(!layers_.empty() && !data_.empty());

		data_[0] = input;
		expected_ = expected;

		forwardPropagate();

		Tensor<T>& output = data_[layers_.size()];

		int outputChannels = output.c_;
		int outputWidth = output.w_;
		int outputHeight = output.h_;

		T loss = 0.0f;

		for (int i = 0; i < output.c_; i++)
			loss += (output[i] - expected_[i]) * (output[i] - expected_[i]);

		/*std::cout << "Loss: " << loss << std::endl;
		std::cout << "Input: " << std::endl << input << std::endl;
		std::cout << "Expected: " << std::endl << expected << std::endl;
		std::cout << "Output: " << std::endl << output << std::endl;
		std::cout << "----------------------------" << std::endl;*/

		return output[0];
	}

public:

	static Builder make(int inputChannels, int inputWidth, int inputHeight, T learningRate = 0.01f)
	{
		return Model::Builder(inputChannels, inputWidth, inputHeight, learningRate);
	}

private:

	void forwardPropagate()
	{
		for (int i = 0; i < layers_.size(); i++)
			layers_[i]->forwardPropagate(data_[i], data_[i + 1]);
	}

	void backwardPropagate()
	{
		// Reset Gradients!
		for (int i = 0; i < gradient_.size(); i++)
			gradient_[i].setTo(0.0f);

		// LOSS
		Tensor<T>& input = data_[0];
		Tensor<T>& output = data_[layers_.size()];
		Tensor<T>& doutput = gradient_[layers_.size()];

		T loss = 0.0f;

		/*for (int i = 0; i < output.c_; i++)
			loss -= expected[i] * log(std::max(0.00001f, output(i, 0, 0)));*/

		for (int i = 0; i < output.c_; i++)
			loss += (output[i] - expected_[i]) * (output[i] - expected_[i]);

		std::cout << "Loss: " << loss << std::endl;
		//std::cout << "Input: " << std::endl << input << std::endl;
		//std::cout << "Expected: " << std::endl << expected_ << std::endl;
		//std::cout << "Output: " << std::endl << output << std::endl;
		//std::cout << "----------------------------" << std::endl;

		/*for (int i = 0; i < doutput.c_; i++)
			doutput(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.001f);*/

		for (int i = 0; i < doutput.c_; i++)
			doutput(i, 0, 0) = 2 * (output[i] - expected_[i]);

		// std::cout << "doutput: " << std::endl << output << std::endl;

		for (int i = layers_.size() - 1; i >= 0; i--)
			layers_[i]->backwardPropagate(data_[i], gradient_[i], data_[i + 1], gradient_[i + 1]);
	}

	void updateParameters()
	{
		for (int i = 0; i < layers_.size(); i++)
			layers_[i]->updateParameters(learningRate_);
	}

private:
	std::vector<std::shared_ptr<Layer<T>>>	layers_;	// Stores the layers with the input at index zero.
	std::vector<Tensor<T>>					data_;		// Stores the input and output of each layer.
	std::vector<Tensor<T>>					gradient_;	// Stores the input and output gradient of each layer.
	Tensor<T>								expected_;	// Stores the expected output for the CURRENT input.

	T learningRate_;

	std::function<void(Tensor<T>&)> inputCallback_;
	std::function<void(Tensor<T>&)> expecCallback_;
};

template <typename T>
class Model<T>::Builder
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
	Builder(int inputChannels, int inputWidth, int inputHeight, T learningRate)
		:  layerCount_(0), learningRate_(learningRate), inpLayer_{ inputChannels, inputWidth, inputHeight }
	{
	}

	Builder& addConvLayer(int kernelSize, int kernelNum)
	{
		convLayers_.push_back({ kernelSize, kernelNum });

		int index = convLayers_.size() - 1;
		std::string name = "conv";

		layerInfo_[layerCount_++] = { name, index };

		return *this;
	}

	Builder& addMaxPoolLayer(int stride);

	Builder& addReluLayer()
	{
		std::string name = "relu";

		layerInfo_[layerCount_++] = { name, {} };

		return *this;
	}

	Builder& addFullyConnectedLayer(int size)
	{
		fcLayers_.push_back({ size });

		int index = fcLayers_.size() - 1;
		std::string name = "fc";

		layerInfo_[layerCount_++] = { name, index };

		return *this;
	}

	Builder& addSoftmaxLayer();

	Model build()
	{
		// Build the network...

		std::vector<Tensor<T>>					data;
		std::vector<Tensor<T>>					gradient;
		std::vector<std::shared_ptr<Layer<T>>>	layers;

		// Input layer...
		data.push_back(Tensor<T>(inpLayer_.inputChannels_, inpLayer_.inputWidth_, inpLayer_.inputHeight_));
		gradient.push_back(Tensor<T>(Tensor<T>(inpLayer_.inputChannels_, inpLayer_.inputWidth_, inpLayer_.inputHeight_)));

		int inputChannels = inpLayer_.inputChannels_;
		int inputWidth = inpLayer_.inputWidth_;
		int inputHeight = inpLayer_.inputHeight_;

		std::cout
			<< "0: input, "
			<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << "), "
			<< std::endl;

		int outputChannels = 0;
		int outputWidth = 0;
		int outputHeight = 0;

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
				data.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<ConvolutionLayer<T>>(cv.kernelSize_, cv.kernelNum_));
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
				data.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<FullyConnectedLayer<T>>(inputSize, fc.size_));
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
				data.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<ReluLayer<T>>());
			}

			inputChannels = outputChannels;
			inputWidth = outputWidth;
			inputHeight = outputHeight;
		}

		return Model(std::move(data), std::move(gradient), std::move(layers), outputChannels, outputWidth, outputHeight, learningRate_);
	}

private:
	int							layerCount_;
	T						learningRate_;
	InpLayer					inpLayer_;
	std::vector<ConvLayer>		convLayers_;
	std::vector<FullConLayer>	fcLayers_;

	std::map<int, std::pair<std::string, int>> layerInfo_;

};