#include "ConvolutionLayer.h"

#include <random>

ConvolutionLayer::ConvolutionLayer(int iN, int iWidth, int iHeight, int kSize, int kNum)
	: 
	input(iN, iWidth, iHeight),
	output(kNum, iWidth - kSize + 1, iHeight - kSize + 1),
	kernel(kNum, kSize, kSize),
	gradin(kNum, kSize, kSize)
{
	for (int n = 0; n < kNum; n++)
		for (int x = 0; x < kSize; x++)
			for (int y = 0; y < kSize; y++)
				kernel(n, x, y) = (float)(rand() % 1000 - 500) / (iN * iWidth * iHeight);
}

ConvolutionLayer::~ConvolutionLayer()
{
}

const Tensor<float>& ConvolutionLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int f = 0; f < kernel.sX; f++)
		for (int i = 0; i < output.sY; i++)
			for (int j = 0; j < output.sZ; j++)
			{
				output(f, i, j) = 0.0f;

				for (int k = 0; k < kernel.sY; k++)
					for (int l = 0; l < kernel.sZ; l++)
						for (int n = 0; n < input.sX; n++)
							output(f, i, j) += input(n, i + k, j + l) * kernel(f, k, l);
			}

	return this->output;
}

void ConvolutionLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	Tensor<float> dw(kernel.sX, kernel.sY, kernel.sZ);

	for (int f = 0; f < kernel.sX; f++)
		for (int i = 0; i < dw.sY; i++)
			for (int j = 0; j < dw.sZ; j++)
			{
				dw(f, i, j) = 0.0f;
				for (int k = 0; k < dout.sY; k++)
					for (int l = 0; l < dout.sZ; l++)
						for (int n = 0; n < input.sX; n++)
						{
							dw(f, i, j) += input(n, i + k, j + l) * dout(f, k, l);
						}

				float delta = dw(f, i, j) * learningRate;
				kernel(f, i, j) -= delta;
			}
}
