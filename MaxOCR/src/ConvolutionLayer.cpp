#include "ConvolutionLayer.h"

#include <random>

#include "Common.h"

ConvolutionLayer::ConvolutionLayer(int iN, int iWidth, int iHeight, int kSize, int kNum)
	: 
	input(iN, iWidth, iHeight),
	output(kNum, iWidth - kSize + 1, iHeight - kSize + 1),
	kernel(kNum, kSize, kSize),
	dinput(kNum, kSize, kSize)
{
	float max = iN * iWidth * iHeight;

	for (int n = 0; n < kNum; n++)
		for (int x = 0; x < kSize; x++)
			for (int y = 0; y < kSize; y++)
				kernel(n, x, y) = 1.0f / max * rand() / float(RAND_MAX);
}

const Tensor<float>& ConvolutionLayer::forwardPropagate(const Tensor<float>& input)
{
	this->input = input;

	for (int f = 0; f < kernel.sX; f++)
		for (int i = 0; i < output.sY; i++)
			for (int j = 0; j < output.sZ; j++)
			{
				float val = 0.0f;
				
				for (int k = 0; k < kernel.sY; k++)
					for (int l = 0; l < kernel.sZ; l++)
						for (int n = 0; n < input.sX; n++)
							val += input(n, i + k, j + l) * kernel(f, k, l);

				output(f, i, j) = val;
			}

	return this->output;
}

void ConvolutionLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	for (int f = 0; f < kernel.sX; f++)
		for (int i = 0; i < kernel.sY; i++)
			for (int j = 0; j < kernel.sZ; j++)
			{
				float grad = 0.0f;

				for (int k = 0; k < dout.sY; k++)
					for (int l = 0; l < dout.sZ; l++)
						for (int n = 0; n < input.sX; n++)
							grad += input(n, i + k, j + l) * dout(f, k, l);

				kernel(f, i, j) -= learningRate * grad;
			}
}
