#pragma once

#include "Common.h"

#include "layers/Layer.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/ReluLayer.h"
#include "layers/SoftmaxLayer.h"

#include "maths/Tensor.h"

class Model
{

friend class Builder;

public:

	void beginEpoch()
	{
		assert(!epochStarted_);

		epochHistory_[epochCount_] = std::vector<double>();
		epochStarted_ = true;
	}

	void endEpoch()
	{
		assert(epochStarted_);

		epochStarted_ = false;

		// Print Epoch Info
		
		double totalLoss{ 0 };
		int iterations = epochHistory_[epochCount_].size();

		for (int i = 0; i < iterations; i++)
			totalLoss += epochHistory_[epochCount_][i];

		double avgLoss = totalLoss / (double)iterations;

		std::cout << "Epoch " << epochCount_ << ", Loss " << avgLoss << std::endl;

		epochCount_++;
	}

	void train(const Tensor<double>& input, const Tensor<double>& expected, double learningRate)
	{
		// TODO: Add my own assert!
		assert(epochStarted_);

		forwardPropagate(input);
		backwardPropagate(input, expected);

		updateParameters(learningRate);

		auto arr = epochHistory_[epochCount_];

		std::cout << "Epoch " << epochCount_ << ", Loss " << arr[arr.size() - 1] << '\r';
	}

	void test(const Tensor<double>& input)
	{
		forwardPropagate(input);

		auto& output = data_[layers_.size()];

		std::cout << "(" << input[0] << ", " << output[0] << "), ";
	}

	static Builder make(int inputChannels, int inputWidth, int inputHeight);

private:

	Model(std::vector<Tensor<double>>&& data, std::vector<Tensor<double>>&& gradient, std::vector<std::shared_ptr<Layer<double>>>&& layers) :
		data_(std::move(data)),
		gradient_(std::move(gradient)),
		layers_(std::move(layers)),
		epochCount_(0),
		epochStarted_(false),
		epochHistory_()
	{
	}

	void forwardPropagate(const Tensor<double>& input)
	{
		layers_[0]->forwardPropagate(input, data_[1]);

		for (int i = 1; i < layers_.size(); i++)
			layers_[i]->forwardPropagate(data_[i], data_[i + 1]);
	}

	void backwardPropagate(const Tensor<double>& input, const Tensor<double>& expected)
	{
		// Reset Gradients!
		for (int i = 0; i < gradient_.size(); i++)
			gradient_[i].set(0.0f);

		Tensor<double>& output = data_[layers_.size()];
		Tensor<double>& doutput = gradient_[layers_.size()];

		double loss = 0.0f;

		/*for (int i = 0; i < output.c_; i++)
			loss -= expected[i] * log(output(i, 0, 0) + 0.0000001);*/

		for (int i = 0; i < output.size_; i++)
			loss += (output[i] - expected[i]) * (output[i] - expected[i]);

		epochHistory_[epochCount_].push_back(loss);

		/*for (int i = 0; i < doutput.c_; i++)
			doutput(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.0000001);*/

		/*std::cout << expected.str() << std::endl;
		std::cout << output.str() << std::endl;
		std::cout << doutput.str() << std::endl;*/

		for (int i = 0; i < output.size_; i++)
			doutput[i] = 2 * (output[i] - expected[i]);

		for (int i = layers_.size() - 1; i > 0; i--)
			layers_[i]->backwardPropagate(data_[i], gradient_[i], data_[i + 1], gradient_[i + 1]);

		// TODO: Try and get around requiring input here...
		layers_[0]->backwardPropagate(input, gradient_[0], data_[1], gradient_[1]);

		// std::cout << loss << std::endl;
	}

	void updateParameters(double learningRate)
	{
		for (int i = 0; i < layers_.size(); i++)
			layers_[i]->updateParameters(learningRate);
	}

private:
	std::vector<std::shared_ptr<Layer<double>>> layers_;	// Stores the layers with the input at index zero.
	
	std::vector<Tensor<double>> data_; // Stores the input and output of each layer.
	std::vector<Tensor<double>> gradient_; // Stores the input and output gradient of each layer.

	int epochCount_;
	bool epochStarted_;
	std::map<int, std::vector<double>> epochHistory_; // Epoch number -> vector of loss
};

class Builder
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
	{}

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

		std::vector<Tensor<double>>					data;
		std::vector<Tensor<double>>					gradient;
		std::vector<std::shared_ptr<Layer<double>>>	layers;

		// Input layer...

		data.push_back(Tensor<double>(inpLayer_.inputChannels_, inpLayer_.inputWidth_, inpLayer_.inputHeight_));
		gradient.push_back(Tensor<double>(Tensor<double>(inpLayer_.inputChannels_, inpLayer_.inputWidth_, inpLayer_.inputHeight_)));

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
				data.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<ConvolutionLayer<double>>(cv.kernelSize_, cv.kernelNum_));
			}
			else if (name == "fc")
			{
				FullConLayer fc = fcLayers_[index];

				outputChannels = 1;
				outputWidth = fc.size_;
				outputHeight = 1;

				int inputSize = inputChannels * inputWidth * inputHeight;

				std::cout
					<< i + 1 << ": fc, "
					<< "size: " << fc.size_ << ", "
					<< "(" << inputChannels << ", " << inputWidth << ", " << inputHeight << ") -> "
					<< "(" << outputChannels << ", " << outputWidth << ", " << outputHeight << ")"
					<< std::endl;

				// Push back the output tensor for this layer...
				data.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<FullyConnectedLayer<double>>(inputSize, fc.size_));
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
				data.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<ReluLayer<double>>());
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
				data.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));
				gradient.push_back(Tensor<double>(Tensor<double>(outputChannels, outputWidth, outputHeight)));

				layers.push_back(std::make_shared<SoftmaxLayer<double>>());
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

Builder Model::make(int inputChannels, int inputWidth, int inputHeight)
{
	return Builder(inputChannels, inputWidth, inputHeight);
}