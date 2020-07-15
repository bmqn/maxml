#include "MaxPoolLayer.h"


MaxPoolLayer::MaxPoolLayer(int iN, int iWidth, int iHeight, int fSize) :
	Layer(Tensor<float>(iN, iWidth, iHeight), Tensor<float>(iN, iWidth / fSize, iHeight / fSize), Tensor<float>(iN, iWidth, iHeight)),
	filterSize(fSize)
{
}

const Tensor<float>& MaxPoolLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int n = 0; n < output.sX; n++)
		for (int i = 0; i < output.sY; i++)
			for (int j = 0; j < output.sZ; j++)
			{
				float maxVal = -INFINITY;

				for (int k = 0; k < filterSize; k++)
					for (int l = 0; l < filterSize; l++)
					{
						float val = input(n, i * (int)filterSize + l, j * (int)filterSize + k);
						if (val > maxVal)
							maxVal = val;
					}

				output(n, i, j) = maxVal;
			}

	return this->output;
}

const Tensor<float>& MaxPoolLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	for (int n = 0; n < output.sX; n++)
		for (int i = 0; i < output.sY; i++)
			for (int j = 0; j < output.sZ; j++)
			{
				float maxVal = -INFINITY;
				int maxK = 0, maxL = 0;

				for (int k = 0; k < filterSize; k++)
					for (int l = 0; l < filterSize; l++)
					{
						float val = input(n, i * filterSize + k, j * filterSize + l);
						if (val > maxVal)
						{
							maxVal = val;
							maxK = k;
							maxL = l;
						}
					}

				dinput(n, i * filterSize + maxK, j * filterSize + maxL) = dout(n, i, j);
			}

	return dinput;
}
