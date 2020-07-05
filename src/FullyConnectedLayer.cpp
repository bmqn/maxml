#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int iN, int iWidth, int iHeight, int oSize)
	:
	input(iN, iWidth, iHeight),
	output(oSize, 1, 1),
	weights(iN * iWidth * iHeight, oSize, 1),
	biases(oSize, 1, 1),
	gradin(iN, iWidth, iHeight)
{
	for (int i = 0; i < iN * iWidth * iHeight; i++)
		for (int j = 0; j < oSize; j++)
			weights(i, j, 0) = (rand() % 1000 - 500) / 100000.0f; // iN * iWidth * iHeight;

	for (int i = 0; i < oSize; i++)
		biases(i, 0, 0) = 0.0f;

	unactivatedOutput = std::vector<float>(oSize);
}

FullyConnectedLayer::~FullyConnectedLayer()
{
}


float activatorFunction(float x)
{
	return tanhf(x);
}

float activatorDerivative(float x)
{
	float th = tanhf(x);
	return 1 - th * th;
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
				{
					int i = input.index(n, x, y);
					sum += input(n, x, y) * weights(i, j, 0);
				}

		sum += biases(j, 0, 0);

		unactivatedOutput[j] = sum;

		// TODO: Add activation function here?
		// output(j, 0, 0) = activatorFunction(sum);
		output(j, 0, 0) = sum;
	}

	return output;
}

const Tensor<float>& FullyConnectedLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	for (int j = 0; j < output.sX; j++)
	{
		// float grad = dout(j, 0, 0) * activatorDerivative(unactivatedOutput[j]);
		float grad = dout(j, 0, 0);

		for (int n = 0; n < input.sX; n++)
			for (int x = 0; x < input.sY; x++)
				for (int y = 0; y < input.sZ; y++)
				{
					int i = input.index(n, x, y);

					weights(i, j, 0) -= learningRate * input(n, x, y) * grad;

					gradin(n, x, y) = weights(i, j, 0) * grad;
				}

		biases(j, 0, 0) -= learningRate * grad;
	}

	return gradin;
}
