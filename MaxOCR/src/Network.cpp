#include "Network.h"

#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/ReluLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"

void Network::setInputLayer(int channels, int width, int height)
{
	if (data_.empty())
	{
		data_.push_back(Tensor<float>(channels, width, height));
	}
	else
	{
		assert(data_[0].c_ == channels && data_[0].w_ == width && data_[0].h_ == height);
	}
}

void Network::addConvLayer(int kernelSize, int kernelNum)
{
	Tensor<float>& input = *getInputData(layers_.size());

	int outputChannels	= kernelNum;
	int outputWidth		= input.w_ - kernelSize + 1;
	int outputHeight	= input.h_ - kernelSize + 1;
	
	layers_.push_back(std::make_shared<ConvolutionLayer>(kernelSize, kernelNum));
	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::addMaxPoolLayer(int stride)
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int outputChannels	= input.c_;
	int outputWidth		= input.w_ / (float) stride;
	int outputHeight	= input.h_ / (float) stride;

	layers_.push_back(std::make_shared<MaxPoolLayer>(stride));
	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::addReluLayer()
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int outputChannels = input.c_;
	int outputWidth = input.w_;
	int outputHeight = input.h_;

	layers_.push_back(std::make_shared<ReluLayer>());
	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::addFullyConnectedLayer(int outputSize)
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int inputSize = input.c_ * input.w_ * input.h_;

	layers_.push_back(std::make_shared<FullyConnectedLayer>(inputSize, outputSize));
	data_.push_back(Tensor<float>(outputSize, 1, 1));
}

void Network::addSoftmaxLayer()
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int outputChannels = input.c_;
	int outputWidth = input.w_;
	int outputHeight = input.h_;

	layers_.push_back(std::make_shared<SoftmaxLayer>());
	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::forwardPropagate(const Tensor<float>& data)
{
	for (int i = 0; i < data.size_; i++)
		data_[0].data_[i] = data.data_[i];

	for (int i = 0; i < layers_.size(); i++)
	{
		layers_[i]->forwardPropagate(*getInputData(i), *getOutputData(i));

		// std::cout << *getOutputData(i);
	}
}

const Tensor<float> Network::getPredictions() const
{
	const Tensor<float>& output = data_[data_.size() - 1];

	Tensor<float> predictions(output.c_, output.w_, output.h_);

	for (int i = 0; i < output.size_; i++)
		predictions[i] = output[i];

	return predictions;
}

const Tensor<float>* Network::getInputData(int layer) const
{
	assert(layer >= 0 && layer < data_.size());

	return &data_[layer];
}

const Tensor<float>* Network::getOutputData(int layer) const
{
	assert(layer >= 0 && layer < data_.size() + 1);

	return &data_[layer + 1];
}

Tensor<float>* Network::getInputData(int layer)
{
	assert(layer >= 0 && layer < data_.size() + 1);

	return &data_[layer];
}

Tensor<float>* Network::getOutputData(int layer)
{
	assert(layer >= 0 && layer < data_.size() + 1);

	return &data_[layer + 1];
}

