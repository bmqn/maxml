#include "ReluLayer.h"

#include "../utils/Activation.h"

#include <algorithm>

void ReluLayer::forwardPropagate(const Tensor<float>& input, Tensor<float>& output)
{
	for (int c = 0; c < input.c_; c++)
		for (int w = 0; w < input.w_; w++)
			for (int h = 0; h < input.h_; h++)
				output(c, w, h) = relu(input(c, w, h));
}

void ReluLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput)
{
	for (int c = 0; c < input.c_; c++)
		for (int w = 0; w < input.w_; w++)
			for (int h = 0; h < input.h_; h++)
				dinput(c, w, h) = reluDerivative(input(c, w, h)) * doutput(c, w, h);
}