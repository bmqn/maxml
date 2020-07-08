#include "SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(int iSize)
	:
	input(iSize, 1, 1),
	output(iSize, 1, 1),
	dinput(iSize, 1, 1)
{
}

const Tensor<float>& SoftmaxLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	float max = -INFINITY;
	float denom = 0.0f;

	for (int i = 0; i < input.sX; i++)
		if (input(i, 0, 0) > max)
			max = input(i, 0, 0);
		
	for (int i = 0; i < input.sX; i++)
		denom += exp(input(i, 0, 0) - max);

	for (int i = 0; i < output.sX; i++)
		output(i, 0, 0) = exp(input(i, 0, 0) - max) / denom;

	return output;
}

const Tensor<float>& SoftmaxLayer::backwardPropagate(const Tensor<float>& dout)
{
	memset(dinput.data.get(), 0.0f, dinput.sX * dinput.sY * dinput.sZ * sizeof(float));

	for (int p = 0; p < input.sX; p++)
		for (int i = 0; i < output.sX; i++)
		{
			if (p == i)
				dinput(p, 0, 0) += dout(i, 0, 0) * output(p, 0, 0) * (1 - output(p, 0, 0));
			else
				dinput(p, 0, 0) += dout(i, 0, 0) * -output(i, 0, 0) * output(p, 0, 0);
		}

	return dinput;
}
