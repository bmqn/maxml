#include "ConvolutionLayer.h"

#include <random>

ConvolutionLayer::ConvolutionLayer(int kernelSize, int kernelNum)
	:
	kernel_(kernelNum, kernelSize, kernelSize),
	dkernel_(kernelNum, kernelSize, kernelSize),
	kernelSize_(kernelSize),
	kernelNum_(kernelNum)
{
	// TODO: Move into seperate header.
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, 1);

	for (int n = 0; n < kernelNum; n++)
		for (int w = 0; w < kernelSize; w++)
			for (int h = 0; h < kernelSize; h++)
				kernel_(n, w, h) = distribution(generator);
}

void ConvolutionLayer::forwardPropagate(const Tensor<float>& input, Tensor<float>& output)
{
	output.setTo(0.0f);

	for (int k = 0; k < kernelNum_; k++)
		for (int c = 0; c < input.c_; c++)
		{
			oneChannelConvolution(&kernel_(k, 0, 0), &input(c, 0, 0), input.w_, input.h_, &output(k, 0, 0));
		}
}

void ConvolutionLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput)
{
	//for (int f = 0; f < dout.sX; f++)
	//	for (int i = 0; i < kernel.sY; i++)
	//		for (int j = 0; j < kernel.sZ; j++)
	//		{
	//			float grad = 0.0f;

	//			for (int k = 0; k < dout.sY; k++)
	//				for (int l = 0; l < dout.sZ; l++)
	//					for (int n = 0; n < input.sX; n++)
	//						grad += input(n, i + k, j + l) * dout(f, k, l);

	//			// kernel(f, i, j) -= learningRate * grad;
	//		}

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
}


void ConvolutionLayer::updateParameters(float learningRate)
{

}

void ConvolutionLayer::oneChannelConvolution(const float* kernel, const float* src, int width, int height, float* dst)
{
	int outputWidth = width - kernelSize_ + 1;
	int outputHeight = height - kernelSize_ + 1;

	for (int w = 0; w < outputWidth; w++)
		for (int h = 0; h < outputHeight; h++)
		{
			float val = 0.0f;

			for (int i = 0; i < kernelSize_; i++)
				for (int j = 0; j < kernelSize_; j++)
				
					val += src[(w + i) * height + (h + j)] * kernel[i * kernelSize_ + j];
				
			dst[w * outputHeight + h] += val;
		}
}