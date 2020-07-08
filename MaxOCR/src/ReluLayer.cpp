#include "ReluLayer.h"

#include <algorithm>

#include "Common.h"

ReluLayer::ReluLayer(int iN, int iWidth, int iHeight)
	:
	input(iN, iWidth, iHeight),
	output(iN, iWidth, iHeight),
	dinput(iN, iWidth, iHeight)
{
}

const Tensor<float>& ReluLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int n = 0; n < input.sX; n++)
		for (int i = 0; i < input.sY; i++)
			for (int j = 0; j < input.sZ; j++)
				output(n, i, j) = relu(input(n, i, j));

	return output;
}

const Tensor<float>& ReluLayer::backwardPropagate(const Tensor<float>& dout)
{
	for (int n = 0; n < input.sX; n++)
		for (int i = 0; i < input.sY; i++)
			for (int j = 0; j < input.sZ; j++)
				dinput(n, i, j) = reluDerivative(input(n, i, j)) * dout(n, i, j);

	return dinput;
}
