#pragma once

#include "../layers/Layer.h"
#include "../layers/ConvolutionLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include "../layers/ReluLayer.h"
#include "../layers/SoftmaxLayer.h"

#include "../utils/Tensor.h"

#include <functional>
#include <map>

template <typename T>
class Model
{
public:
	class Builder;

	Model(std::vector<Tensor<T>>&& data, std::vector<Tensor<T>>&& gradient, std::vector<std::shared_ptr<Layer<T>>>&& layers) : 
		data_(std::move(data)),
		gradient_(std::move(gradient)),
		layers_(std::move(layers)),
		epochCount_(0),
		epochStarted_(false),
		epochHistory_()
	{
	}

	void beginEpoch()
	{
		assert(!epochStarted_);

		epochHistory_[epochCount_] = std::vector<T>();
		epochStarted_ = true;
	}

	void endEpoch()
	{
		assert(epochStarted_);

		epochStarted_ = false;

		// Print Epoch Info
		
		T totalLoss{ 0 };
		int iterations = epochHistory_[epochCount_].size();

		for (int i = 0; i < iterations; i++)
			totalLoss += epochHistory_[epochCount_][i];

		T avgLoss = totalLoss / (T)iterations;

		std::cout << "Epoch " << epochCount_ << ", Loss " << avgLoss << std::endl;

		epochCount_++;
	}

	void train(const Tensor<T>& input, const Tensor<T>& expected, T learningRate)
	{
		// TODO: Add my own assert!
		assert(epochStarted_);

		forwardPropagate(input);
		backwardPropagate(expected);

		updateParameters(learningRate);
	}

	Tensor<T> test(const Tensor<T>& input)
	{
		forwardPropagate(input);

		Tensor<T>& output = data_[layers_.size()];

		return output;
	}

	T getLoss()
	{
		auto arr = epochHistory_[epochCount_];
		return arr[arr.size() - 1];
	}

public:

	static Builder make(int inputChannels, int inputWidth, int inputHeight)
	{
		return Model::Builder(inputChannels, inputWidth, inputHeight);
	}

private:

	void forwardPropagate(const Tensor<T>& input)
	{
		layers_[0]->forwardPropagate(input, data_[1]);

		for (int i = 1; i < layers_.size(); i++)
			layers_[i]->forwardPropagate(data_[i], data_[i + 1]);
	}

	void backwardPropagate(const Tensor<T>& expected)
	{
		// Reset Gradients!
		for (int i = 0; i < gradient_.size(); i++)
			gradient_[i].setTo(0.0f);

		Tensor<T>& output = data_[layers_.size()];
		Tensor<T>& doutput = gradient_[layers_.size()];

		T loss = 0.0f;

		for (int i = 0; i < output.c_; i++)
			loss -= expected[i] * log(output(i, 0, 0) + 0.0000001);

		/*for (int i = 0; i < output.c_; i++)
			loss += (output[i] - expected[i]) * (output[i] - (expected)[i]);*/

		epochHistory_[epochCount_].push_back(loss);

		for (int i = 0; i < doutput.c_; i++)
			doutput(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.0000001);

		/*for (int i = 0; i < doutput.c_; i++)
			doutput(i, 0, 0) = 2 * (output[i] - expected[i]);*/

		for (int i = layers_.size() - 1; i >= 0; i--)
			layers_[i]->backwardPropagate(data_[i], gradient_[i], data_[i + 1], gradient_[i + 1]);

		// std::cout << loss << std::endl;
	}

	void updateParameters(T learningRate)
	{
		for (int i = 0; i < layers_.size(); i++)
			layers_[i]->updateParameters(learningRate);
	}

private:
	std::vector<std::shared_ptr<Layer<T>>> layers_;	// Stores the layers with the input at index zero.
	
	std::vector<Tensor<T>> data_;		// Stores the input and output of each layer.
	std::vector<Tensor<T>> gradient_;	// Stores the input and output gradient of each layer.

	int epochCount_;
	bool epochStarted_;
	std::map<int, std::vector<T>> epochHistory_; // Epoch number -> vector of loss
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
	Builder(int inputChannels, int inputWidth, int inputHeight) :
		layerCount_(0), inpLayer_{ inputChannels, inputWidth, inputHeight }
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

	Builder& addSoftmaxLayer()
	{
		std::string name = "softmax";

		layerInfo_[layerCount_++] = { name, {} };

		return *this;
	}

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
			else if (name == "softmax")
			{
				outputChannels = inputChannels;
				outputWidth = inputWidth;
				outputHeight = inputHeight;

				std::cout
					<< i + 1 << ": softmax, "
					<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << ") -> "
					<< "(" << outputChannels << ", " << outputWidth << ", " << outputHeight << ")"
					<< std::endl;

				// Push back the output tensor for this layer...
				data.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<T>(Tensor<T>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<SoftmaxLayer<T>>());
			}

			inputChannels = outputChannels;
			inputWidth = outputWidth;
			inputHeight = outputHeight;
		}

		return Model(std::move(data), std::move(gradient), std::move(layers));
	}

private:
	int layerCount_;
	InpLayer inpLayer_;
	std::vector<ConvLayer> convLayers_;
	std::vector<FullConLayer> fcLayers_;

	std::map<int, std::pair<std::string, int>> layerInfo_;

};