#include "FullyConnectedLayer.h"

#include <random>

#include "Common.h"


FullyConnectedLayer::FullyConnectedLayer(int iSize, int oSize)
	:
	Layer(Tensor<float>(iSize, 1, 1), Tensor<float>(oSize, 1, 1), Tensor<float>(iSize, 1, 1)),
	weights(iSize, oSize, 1),
	biases(oSize, 1, 1)
{
	for (int i = 0; i < iSize; i++)
		for (int j = 0; j < oSize; j++)
			weights(i, j, 0) = 2.19722f / iSize * rand() / float(RAND_MAX);

	for (int i = 0; i < oSize; i++)
		biases(i, 0, 0) = 0.0f;
}

const Tensor<float>& FullyConnectedLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int j = 0; j < output.sX; j++)
	{
		float sum = 0;

		for (int i = 0; i < input.sX; i++)
			sum += input(i, 0, 0) * weights(i, j, 0);

		sum += biases(j, 0, 0);

		output(j, 0, 0) = sum;
	}

	return output;
}

const Tensor<float>& FullyConnectedLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	memset(dinput.data.get(), 0.0f, dinput.sX * dinput.sY * dinput.sZ * sizeof(float));

	for (int j = 0; j < output.sX; j++)
	{	
		for (int i = 0; i < input.sX; i++)
		{
			weights(i, j, 0) -= learningRate * dout(j, 0, 0) * input(i, 0, 0);

			dinput(i, 0, 0) += dout(j, 0, 0) * weights(i, j, 0);
		}

		biases(j, 0, 0) -= learningRate * dout(j, 0, 0);
	}

	return dinput;
}
