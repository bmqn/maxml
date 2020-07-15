#include "FlattenLayer.h"

FlattenLayer::FlattenLayer(int iN, int iWidth, int iHeight)
	:
	Layer(Tensor<float>(iN, iWidth, iHeight), Tensor<float>(iN* iWidth* iHeight, 1, 1), Tensor<float>(iN, iWidth, iHeight))
{
}

const Tensor<float>& FlattenLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int n = 0; n < input.sX; n++)
		for (int i = 0; i < input.sY; i++)
			for (int j = 0; j < input.sZ; j++)
			{
				int index = input.index(n, i, j);
				output(index, 0, 0) = input(n, i, j);
			}

	return output;
}

const Tensor<float>& FlattenLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	for (int n = 0; n < input.sX; n++)
		for (int i = 0; i < input.sY; i++)
			for (int j = 0; j < input.sZ; j++)
			{
				int index = dinput.index(n, i, j);
				dinput(n, i, j) = dout(index, 0, 0);
			}

	return dinput;
}
