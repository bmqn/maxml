#include "SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(int iSize)
	:
	input(iSize, 1, 1),
	output(iSize, 1, 1),
	gradin(iSize, 1, 1)
{
}

SoftmaxLayer::~SoftmaxLayer()
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

const Tensor<float>& SoftmaxLayer::backwardPropagate(const Tensor<float>& dL_dy)
{
	Tensor<float> dy_dz(output.sX, input.sX, 1);

	for (int i = 0; i < output.sX; i++)
		for (int l = 0; l < input.sX; l++)
		{
			if (i != l)
				dy_dz(i, l, 0) = -output(i, 0, 0) * output(l, 0, 0);
			else
				dy_dz(i, l, 0) = output(i, 0, 0) * ( 1 - output(l, 0, 0));
		}

	for (int l = 0; l < input.sX; l++)
	{
		gradin(l, 0, 0) = 0.0f;

		for (int i = 0; i < output.sX; i++)
			gradin(l, 0, 0) += dL_dy(i, 0, 0) * dy_dz(i, l, 0);
	}

	return gradin;
}
