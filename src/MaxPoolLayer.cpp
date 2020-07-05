#include "MaxPoolLayer.h"


MaxPoolLayer::MaxPoolLayer(int iN, int iWidth, int iHeight, int fSize) :
	filterSize(fSize),
	input(iN, iWidth, iHeight),
	output(iN, iWidth / fSize, iHeight / fSize),
	gradin(iN, iWidth, iHeight)
{
}

MaxPoolLayer::~MaxPoolLayer()
{
}

const Tensor<float>& MaxPoolLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int i = 0; i < output.sY; i++)
	{
		for (int j = 0; j < output.sZ; j++)
		{
			for (int n = 0; n < output.sX; n++)
			{
				float maxVal = -INFINITY;

				for (int k = 0; k < filterSize; k++)
				{
					for (int l = 0; l < filterSize; l++)
					{
						float val = input(n, i * filterSize + l, j * filterSize + k);
						if (val > maxVal)
							maxVal = val;
					}
				}

				output(n, i, j) = maxVal;
			}
		}
	}

	return this->output;
}

const Tensor<float>& MaxPoolLayer::backwardPropagate(const Tensor<float>& dout)
{

	for (int i = 0; i < gradin.sX; i++)
		for (int j = 0; j < gradin.sY; j++)
			for (int k = 0; k < gradin.sZ; k++)
				gradin(i, j, k) = 0.0f;

	for (int i = 0; i < output.sY; i++)
	{
		for (int j = 0; j < output.sZ; j++)
		{
			for (int n = 0; n < output.sX; n++)
			{
				int maxK = 0, maxL = 0;
				float maxVal = -INFINITY;

				for (int k = 0; k < filterSize; k++)
				{
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
				}
				gradin(n, i * filterSize + maxK, j * filterSize + maxL) = dout(n, i, j);
			}
		}
	}

	return gradin;
}
