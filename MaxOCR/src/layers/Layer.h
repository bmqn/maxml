#pragma once

#include "../Tensor.h"

class Layer
{
public:
	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) = 0;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) = 0;

	Layer(Tensor<float> input, Tensor<float> output, Tensor<float> dinput)
		: input(input), output(output), dinput(dinput)
	{}

public:
	Tensor<float> input;
	Tensor<float> output;

	Tensor<float> dinput;
};

