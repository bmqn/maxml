#pragma once

#include "Layer.h"

template<typename T>
class ReluLayer : public Layer<T>
{

public:
	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override
	{
		for (int s = 0; s < input.size_; s++)
			output[s] = input[s] < 0 ? 0 : input[s];
	}

	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override
	{
		for (int s = 0; s < input.size_; s++)
			dinput[s] = (input[s] < 0 ? 0 : 1) * doutput[s];
	}

	virtual void updateParameters(T learningRate) override {}
};

