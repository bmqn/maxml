#include "Network.h"

#include "layers/InputLayer.h"
#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/ReluLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"

// Requires complete defn...
Network::Builder Network::make(int channels, int width, int height, float learningRate)
{
	return Network::Builder(channels, width, height, learningRate);
}


void Network::train()
{
	assert(inputCallback_ && expecCallback_);

	inputCallback_(data_[0]);

	forwardPropagate();
	backwardPropagate();
	updateParameters();
}

Tensor<float> Network::predict(const Tensor<float>& input)
{
	memcpy(data_[0].data_, input.data_, 1 * sizeof(float));

	forwardPropagate();

	Tensor<float> output(1, 1, 1);
	memcpy(output.data_, data_[layers_.size()].data_, 1 * sizeof(float));

	return output;
}

void Network::forwardPropagate()
{
	for (int i = 0; i < layers_.size(); i++)
	{
		layers_[i]->forwardPropagate(data_[i], data_[i + 1]);

		// std::cout << "Input: " << std::endl << data_[i];
		// std::cout << "Output: " << std::endl << data_[i + 1];
	}
}

void Network::backwardPropagate()
{
	// Reset Gradients!
	for (int i = 0; i < gradient_.size(); i++)
		gradient_[i].setTo(0.0f);

	expected_ = std::make_shared<Tensor<float>>(1, 1, 1);

	expecCallback_(*expected_.get());

	// LOSS
	Tensor<float>& input = data_[0];
	Tensor<float>& output = data_[layers_.size()];
	Tensor<float>& doutput = gradient_[layers_.size()];

	int outputChannels	= output.c_;
	int outputWidth		= output.w_;
	int outputHeight	= output.h_;

	float loss = 0.0f;

	/*for (int i = 0; i < output.c_; i++)
		loss -= expected[i] * log(std::max(0.00001f, output(i, 0, 0)));*/

	for (int i = 0; i < output.c_; i++)
		loss += (output[i] - expected_->operator[](i)) * (output[i] - expected_->operator[](i));

	/*std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "Input: " << std::endl << input << std::endl;
	std::cout << "Expected: " << std::endl << expected_->str() << std::endl;
	std::cout << "Output: " << std::endl << output << std::endl;*/
	std::cout << "Loss: " << loss << std::endl;
	//std::cout << "--------------------------------------------------" << std::endl;

	/*for (int i = 0; i < doutput.c_; i++)
		doutput(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.001f);*/

	for (int i = 0; i < doutput.c_; i++)
		doutput(i, 0, 0) = 2 * (output[i] - expected_->operator[](i));

	for (int i = layers_.size() - 1; i >= 0; i--)
	{
		// std::cout << "Output: " << std::endl << data_[i + 1];

		layers_[i]->backwardPropagate(data_[i], gradient_[i], data_[i + 1], gradient_[i + 1]);
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

//const Tensor<float>* Network::getInputData(int layer) const
//{
//	assert(layer >= 0 && layer < data_.size());
//
//	return &data_[layer];
//}
//
//const Tensor<float>* Network::getOutputData(int layer) const
//{
//	assert(layer >= 0 && layer + 1 < data_.size());
//
//	return &data_[layer + 1];
//}
//
//Tensor<float>* Network::getInputData(int layer)
//{
//	assert(layer >= 0 && layer < data_.size());
//
//	return &data_[layer];
//}
//
//Tensor<float>* Network::getOutputData(int layer)
//{
//	assert(layer >= 0 && layer + 1 < data_.size());
//
//	return &data_[layer + 1];
//}
//
//const Tensor<float>* Network::getInputGradient(int layer) const
//{
//	assert(layer >= 0 && layer < data_.size());
//
//	return &gradient_[layer];
//}
//
//const Tensor<float>* Network::getOutputGradient(int layer) const
//{
//	assert(layer >= 0 && layer + 1 < data_.size());
//
//	return &gradient_[layer + 1];
//}
//
//Tensor<float>* Network::getInputGradient(int layer)
//{
//	assert(layer >= 0 && layer < data_.size() + 1);
//
//	return &gradient_[layer];
//}
//
//Tensor<float>* Network::getOutputGradient(int layer)
//{
//	assert(layer >= 0 && layer + 1 < data_.size());
//
//	return &gradient_[layer + 1];
//}