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
	/*for (int n = 0; n < input.sX; n++)
		for (int i = 0; i < input.sY; i++)
			for (int j = 0; j < input.sZ; j++)
				dinput(n, i, j) = reluDerivative(input(n, i, j)) * dout(n, i, j);

	return dinput;*/
}

void ReluLayer::updateParameters(float learningRate)
{

}
