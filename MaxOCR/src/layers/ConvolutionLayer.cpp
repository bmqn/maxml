#include "ConvolutionLayer.h"

#include <random>
#include "../Common.h"

ConvolutionLayer::ConvolutionLayer(int iN, int iWidth, int iHeight, int kSize, int kNum)
	:
	Layer(Tensor<float>(iN, iWidth, iHeight), Tensor<float>(kNum, iWidth - kSize + 1, iHeight - kSize + 1), Tensor<float>(iN, iWidth, iHeight)),
	kernel(kNum, kSize, kSize)
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

const Tensor<float>& ConvolutionLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
{
	for (int f = 0; f < dout.sX; f++)
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

	//const int dx = kernel.sY / 2;
	//const int dy = kernel.sZ / 2;

	//Tensor<float> rotated_kernel(kernel.sX, kernel.sY, kernel.sZ);

	//for (int f = 0; f < kernel.sX; f++)
	//	for (int i = kernel.sY - 1; i >= 0; i--)
	//		for (int j = kernel.sZ - 1; j >= 0; j--)
	//		{
	//			float val = kernel(f, i, j);
	//			rotated_kernel(f, kernel.sY - i - 1, kernel.sZ - j - 1) = val;
	//		}

	//for (int f = 0; f < kernel.sX; f++)
	//	for (int i = 0; i < dinput.sY; i++)
	//		for (int j = 0; j < dinput.sZ; j++)
	//		{
	//			float grad = 0.0f;

	//			for (int k = 0; k < kernel.sY; k++)
	//				for (int l = 0; l < kernel.sZ; l++)
	//					for (int n = 0; n < dout.sX; n++)
	//						grad += dout(n, i + k - dx, j + l - dy) * rotated_kernel(f, k, l);

	//			dinput(f, i, j) = grad;
	//		}

	// std::cout << dinput;

	return dinput;
}
