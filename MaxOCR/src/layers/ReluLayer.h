#pragma once

#include "Layer.h"

#include "../utils/Activation.h"

template<typename T>
class ReluLayer : public Layer<T>
{

public:
	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override
	{
		for (int c = 0; c < input.c_; c++)
			for (int w = 0; w < input.w_; w++)
				for (int h = 0; h < input.h_; h++)
					output(c, w, h) = relu(input(c, w, h));
	}

	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override
	{
		for (int c = 0; c < input.c_; c++)
			for (int w = 0; w < input.w_; w++)
				for (int h = 0; h < input.h_; h++)
					dinput(c, w, h) = reluDerivative(input(c, w, h)) * doutput(c, w, h);
	}

	virtual void updateParameters(T learningRate) override {}
};

