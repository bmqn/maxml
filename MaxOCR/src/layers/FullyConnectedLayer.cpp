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
	/*std::cout << "weights" << std::endl << weights_ << std::endl;
	std::cout << "biases" << std::endl << biases_ << std::endl;*/

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

	/*std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "weights: " << std::endl << weights_ << std::endl;
	std::cout << "biases: " << std::endl << biases_ << std::endl;
	std::cout << "input: " << std::endl << input << std::endl;
	std::cout << "output: " << std::endl << output << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;*/
}

void FullyConnectedLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput)
{
	dweights_.setTo(0.0f);
	dbiases_.setTo(0.0f);

	for (int j = 0; j < outputSize_; j++)
	{
		float factor = doutput(j, 0, 0);

		for (int i = 0; i < inputSize_; i++)
			dweights_(0, j, i) += factor * input[i];

		dbiases_(0, 0, j) += factor;

		for (int i = 0; i < inputSize_; i++)
			dinput[i] += factor * weights_(0, j, i);
	}

	/*std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "weights: "	<< std::endl << weights_ << std::endl;
	std::cout << "biases: "		<< std::endl << biases_ << std::endl;
	std::cout << "dweights: "	<< std::endl << dweights_ << std::endl;
	std::cout << "dbiases: "	<< std::endl << dbiases_ << std::endl;
	std::cout << "donput: "		<< std::endl << dinput << std::endl;
	std::cout << "doutput: "	<< std::endl << doutput << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;*/

	/*for (int j = 0; j < output.c_; j++)
	{	
		for (int i = 0; i < input.c_; i++)
		{
			dweights_(0, i, j) = doutput(j, 0, 0) * input(i, 0, 0);

			dinput(i, 0, 0) += doutput(j, 0, 0) * weights_(0, i, j);
		}

		dbiases_(j, 0, 0) = doutput(j, 0, 0);
	}*/
}

void FullyConnectedLayer::updateParameters(float learningRate)
{
	/*std::cout << "dweights" << std::endl << dweights_ << std::endl;
	std::cout << "weights" << std::endl << weights_ << std::endl;

	std::cout << "dbiases" << std::endl << dbiases_ << std::endl;
	std::cout << "biases" << std::endl << biases_ << std::endl;*/

	for (int s = 0; s < dbiases_.size_; s++)
		weights_[s] -= learningRate * dweights_[s];

	for (int s = 0; s < dbiases_.size_; s++)
		biases_[s] -= learningRate * dbiases_[s];
}
