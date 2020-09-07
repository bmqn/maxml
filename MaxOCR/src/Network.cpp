#include "Network.h"

#include "layers/InputLayer.h"
#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/ReluLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"

void Network::addInputLayer(int channels, int width, int height)
{
	assert(layers_.empty());

	data_.push_back(Tensor<float>(channels, width, height));
	gradient_.push_back(Tensor<float>(channels, width, height));

	int outputChannels	= channels;
	int outputWidth		= width;
	int outputHeight	= height;

	layers_.push_back(std::make_shared<InputLayer>());

	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
	gradient_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::addConvLayer(int kernelSize, int kernelNum)
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int outputChannels	= kernelNum;
	int outputWidth		= input.w_ - kernelSize + 1;
	int outputHeight	= input.h_ - kernelSize + 1;
	
	layers_.push_back(std::make_shared<ConvolutionLayer>(kernelSize, kernelNum));

	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
	gradient_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
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
	gradient_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::addReluLayer()
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int outputChannels	= input.c_;
	int outputWidth		= input.w_;
	int outputHeight	= input.h_;

	layers_.push_back(std::make_shared<ReluLayer>());

	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
	gradient_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
}

void Network::addFullyConnectedLayer(int outputSize)
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size());

	int inputSize = input.c_ * input.w_ * input.h_;

	layers_.push_back(std::make_shared<FullyConnectedLayer>(inputSize, outputSize));

	data_.push_back(Tensor<float>(outputSize, 1, 1));
	gradient_.push_back(Tensor<float>(outputSize, 1, 1));
}

void Network::addSoftmaxLayer()
{
	assert(!data_.empty());

	Tensor<float>& input = *getInputData(layers_.size() - 1);

	int outputChannels	= input.c_;
	int outputWidth		= input.w_;
	int outputHeight	= input.h_;

	layers_.push_back(std::make_shared<SoftmaxLayer>());

	data_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));
	gradient_.push_back(Tensor<float>(outputChannels, outputWidth, outputHeight));

}

void Network::forwardPropagate()
{
	assert(dataCallback_, "You must set the data callback!");

	// Set input.
	input_ = &data_[0];

	std::pair<Tensor<float>*, Tensor<float>*> pair = { input_, &expected_ };

	// Fetch Data
	dataCallback_(pair);

	for (int i = 0; i < layers_.size(); i++)
	{
		layers_[i]->forwardPropagate(*getInputData(i), *getOutputData(i));
	}
}

void Network::backwardPropagate()
{
	// LOSS
	Tensor<float>& input = *getInputData(0);
	Tensor<float>& output = *getOutputData(layers_.size() - 1);
	Tensor<float>& doutput = *getOutputGradient(layers_.size() - 1);

	int outputChannels	= output.c_;
	int outputWidth		= output.w_;
	int outputHeight	= output.h_;

	float loss = 0.0f;

	/*for (int i = 0; i < output.c_; i++)
		loss -= expected[i] * log(std::max(0.00001f, output(i, 0, 0)));*/

	for (int i = 0; i < output.c_; i++)
		loss += (output[i] - expected_.operator[](i)) * (output[i] - expected_.operator[](i));

	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "Input: " << std::endl <<input << std::endl;
	std::cout << "Expected: " << std::endl << expected_.str() << std::endl;
	std::cout << "Output: " << std::endl << output << std::endl;
	std::cout << "Loss: " << loss << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;

	/*for (int i = 0; i < doutput.c_; i++)
		doutput(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.001f);*/

	for (int i = 0; i < doutput.c_; i++)
		doutput(i, 0, 0) = 2 * (output[i] - expected_.operator[](i));

	for (int i = layers_.size() - 1; i >= 0; i--)
	{
		layers_[i]->backwardPropagate(*getInputData(i), *getInputGradient(i), *getOutputData(i), *getOutputGradient(i));
	}
}

void Network::updateParameters()
{
	float learningRate = 0.01f;

	for (int i = 0; i < layers_.size(); i++)
	{
		layers_[i]->updateParameters(learningRate);
	}
}

const Tensor<float>* Network::getInputData(int layer) const
{
	assert(layer >= 0 && layer < data_.size());

	return &data_[layer];
}

const Tensor<float>* Network::getOutputData(int layer) const
{
	assert(layer >= 0 && layer + 1 < data_.size());

	return &data_[layer + 1];
}

Tensor<float>* Network::getInputData(int layer)
{
	assert(layer >= 0 && layer < data_.size());

	return &data_[layer];
}

Tensor<float>* Network::getOutputData(int layer)
{
	assert(layer >= 0 && layer + 1 < data_.size());

	return &data_[layer + 1];
}

const Tensor<float>* Network::getInputGradient(int layer) const
{
	assert(layer >= 0 && layer < data_.size());

	return &gradient_[layer];
}

const Tensor<float>* Network::getOutputGradient(int layer) const
{
	assert(layer >= 0 && layer + 1 < data_.size());

	return &gradient_[layer + 1];
}

Tensor<float>* Network::getInputGradient(int layer)
{
	assert(layer >= 0 && layer < data_.size() + 1);

	return &gradient_[layer];
}

Tensor<float>* Network::getOutputGradient(int layer)
{
	assert(layer >= 0 && layer + 1 < data_.size());

	return &gradient_[layer + 1];
}

