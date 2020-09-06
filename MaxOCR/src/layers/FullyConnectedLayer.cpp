#include "FullyConnectedLayer.h"

#include <random>


FullyConnectedLayer::FullyConnectedLayer(int inputSize, int outputSize)
	:
	weights_(1, outputSize, inputSize),
	biases_(1, 1, outputSize),
	dweights_(1, outputSize, inputSize),
	dbiases_(1, 1, outputSize),
	inputSize_(inputSize),
	outputSize_(outputSize)
{
	// TODO: Move into seperate header.
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, 1);

	for (int i = 0; i < outputSize; i++)
		for (int j = 0; j < inputSize; j++)
			weights_(0, i, j) = distribution(generator);

	for (int i = 0; i < outputSize; i++)
		biases_(0, 0, i) = 0.0f;
}

void FullyConnectedLayer::forwardPropagate(const Tensor<float>& input, Tensor<float>& output)
{
	for (int j = 0; j < outputSize_; j++)
		{
			float val = 0.0f;

			for (int i = 0; i < inputSize_; i++)
			{
				// std::cout << input[i] << ", " << (&weights_(0, j, 0))[i] << std::endl;
				val += input[i] * (&weights_(0, j, 0))[i];

			}

			output(j, 0, 0) = val + biases_(0, 0, j);
		}
}

void FullyConnectedLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput)
{
	/*for (int j = 0; j < output.c_; j++)
	{	
		for (int i = 0; i < input.c_; i++)
		{
			dweights(i, j, 0) = doutput(j, 0, 0) * input(i, 0, 0);

			dinput(i, 0, 0) += doutput(j, 0, 0) * weights(i, j, 0);
		}

		dbiases(j, 0, 0) = doutput(j, 0, 0);
	}*/
}

void FullyConnectedLayer::updateParameters(float learningRate)
{

}
