#pragma once

#include "Layer.h"

template<typename T>
class FullyConnectedLayer : public Layer<T>
{

public:
	FullyConnectedLayer() = delete;

	FullyConnectedLayer(int inputSize, int numNeurons) :
		weights_(1, numNeurons, inputSize),
		biases_(1, numNeurons, 1),
		dweights_(1, numNeurons, inputSize),
		dbiases_(1, numNeurons, 1)
	{
		// TODO: Move into seperate header.
		std::default_random_engine generator;
		std::normal_distribution<T> distribution(0, 1);

		for (int i = 0; i < weights_.size_; i++)
			weights_[i] = distribution(generator);

		for (int i = 0; i < biases_.size_; i++)
			biases_[i] = 0;
	}

	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override
	{
		op::add(output, op::matmul(weights_, input), biases_);
	}

	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override
	{
		op::matmul(dinput, op::transpose(weights_), doutput);

		op::matmul(dweights_, doutput, op::transpose(input));
		op::copy(dbiases_, doutput);
	}

	virtual void updateParameters(T learningRate) override
	{
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
};