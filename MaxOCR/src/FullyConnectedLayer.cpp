#include "FullyConnectedLayer.h"

#include <random>

#include "Common.h"


FullyConnectedLayer::FullyConnectedLayer(int iN, int iWidth, int iHeight, int oSize)
	:
	input(iN, iWidth, iHeight),
	output(oSize, 1, 1),
	weights(iN * iWidth * iHeight, oSize, 1),
	biases(oSize, 1, 1),
	dinput(iN, iWidth, iHeight)
{
	float max = iN * iWidth * iHeight;

	for (int i = 0; i < iN * iWidth * iHeight; i++)
		for (int j = 0; j < oSize; j++)
			weights(i, j, 0) = 2.19722f / max * rand() / float(RAND_MAX);

	for (int i = 0; i < oSize; i++)
		biases(i, 0, 0) = 0.0f;
}

const Tensor<float>& FullyConnectedLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int j = 0; j < output.sX; j++)
	{
		float sum = 0;

		for (int n = 0; n < input.sX; n++)
			for (int x = 0; x < input.sY; x++)
				for (int y = 0; y < input.sZ; y++)
					sum += input(n, x, y) * weights(input.index(n, x, y), j, 0);

		sum += biases(j, 0, 0);

		output(j, 0, 0) = sum;
	}

	return output;
}

const Tensor<float>& FullyConnectedLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	memset(dinput.data, 0.0f, dinput.sX * dinput.sY * dinput.sZ * sizeof(float));

	for (int j = 0; j < output.sX; j++)
	{	
		for (int n = 0; n < input.sX; n++)
			for (int x = 0; x < input.sY; x++)
				for (int y = 0; y < input.sZ; y++)
				{
					int i = input.index(n, x, y);

					weights(i, j, 0) -= learningRate * dout(j, 0, 0) * input(n, x, y);

					dinput(n, x, y) += dout(j, 0, 0) * weights(i, j, 0);
				}

		biases(j, 0, 0) -= learningRate * dout(j, 0, 0);
	}

	return dinput;
}
