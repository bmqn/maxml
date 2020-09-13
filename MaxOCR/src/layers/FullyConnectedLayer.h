#pragma once

#include "Layer.h"

template<typename T>
class FullyConnectedLayer : public Layer<T>
{

public:
	FullyConnectedLayer() = delete;

	FullyConnectedLayer(int inputSize, int outputSize)
		: weights_(1, outputSize, inputSize), biases_(1, 1, outputSize), dweights_(1, outputSize, inputSize),
			dbiases_(1, 1, outputSize), inputSize_(inputSize), outputSize_(outputSize)
	{
		// TODO: Move into seperate header.
		std::default_random_engine generator;
		std::normal_distribution<T> distribution(0, 0.1);

		generator.seed(time(NULL));

		for (int i = 0; i < outputSize; i++)
			for (int j = 0; j < inputSize; j++)
				weights_(0, i, j) = distribution(generator);

		for (int i = 0; i < outputSize; i++)
			biases_(0, 0, i) = 0.0f;
	}

	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override
	{
		// std::cout << "--------------------------------------------------" << std::endl;

		for (int j = 0; j < outputSize_; j++)
		{
			T val = 0.0f;

			for (int i = 0; i < inputSize_; i++)
			{
				val += input[i] * (&weights_(0, j, 0))[i];
				// std::cout << input[i] << ", " << (&weights_(0, j, 0))[i] << std::endl;
			}

			output(j, 0, 0) = val + biases_(0, 0, j);
		}


		/*std::cout << "weights: " << std::endl << weights_ << std::endl;
		std::cout << "biases: " << std::endl << biases_ << std::endl;
		std::cout << "input: " << std::endl << input << std::endl;
		std::cout << "output: " << std::endl << output << std::endl;
		std::cout << "--------------------------------------------------" << std::endl;*/
	}

	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override
	{
		dweights_.setTo(0.0f);
		dbiases_.setTo(0.0f);

		for (int j = 0; j < outputSize_; j++)
		{
			T factor = doutput(j, 0, 0);

			for (int i = 0; i < inputSize_; i++)
				dweights_(0, j, i) += factor * input[i];

			dbiases_(0, 0, j) += factor;

			for (int i = 0; i < inputSize_; i++)
				dinput[i] += factor * weights_(0, j, i);
		}

		/*std::cout << "input" << std::endl << input << std::endl;
		std::cout << "dinput" << std::endl << dinput << std::endl;
		std::cout << "output" << std::endl << output << std::endl;
		std::cout << "doutput" << std::endl << doutput << std::endl;*/

		/*std::cout << "weights: "	<< std::endl << weights_ << std::endl;
		std::cout << "biases: "		<< std::endl << biases_ << std::endl;
		std::cout << "input: " << std::endl << input << std::endl;
		std::cout << "output: " << std::endl << output << std::endl;
		std::cout << "dinput: " << std::endl << dinput << std::endl;
		std::cout << "doutput: " << std::endl << doutput << std::endl;
		std::cout << "dweights: "	<< std::endl << dweights_ << std::endl;
		std::cout << "dbiases: "	<< std::endl << dbiases_ << std::endl;
		std::cout << "--------------------------------------------------" << std::endl;*/
	}

	virtual void updateParameters(T learningRate) override
	{
		/*std::cout << "dweights" << std::endl << dweights_ << std::endl;
		std::cout << "weights" << std::endl << weights_ << std::endl;

		std::cout << "dbiases" << std::endl << dbiases_ << std::endl;
		std::cout << "biases" << std::endl << biases_ << std::endl;*/

		for (int s = 0; s < dweights_.size_; s++)
			weights_[s] -= learningRate * dweights_[s];

		for (int s = 0; s < dbiases_.size_; s++)
			biases_[s] -= learningRate * dbiases_[s];
	}

private:
	Tensor<T>		weights_;
	Tensor<T>		biases_;

	Tensor<T>		dweights_;
	Tensor<T>		dbiases_;
	int				inputSize_;
	int				outputSize_;

};