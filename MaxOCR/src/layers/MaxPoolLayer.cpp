#include "MaxPoolLayer.h"


MaxPoolLayer::MaxPoolLayer(int stride)
	: stride_(stride)
{
}

void MaxPoolLayer::forwardPropagate(const Tensor<float>& input, Tensor<float>& output)
{
	for (int c = 0; c < output.c_; c++)
		for (int w = 0; w < output.w_; w++)
			for (int h = 0; h < output.h_; h++)
			{
				float maxVal = input(c, w * stride_, h * stride_);

				for (int i = 0; i < stride_; i++)
					for (int j = 0; j < stride_; j++)
					{
						float val = input(c, w * stride_ + i, h * stride_ + j);

						if (val > maxVal)
							maxVal = val;
					}

				output(c, w, h) = maxVal;
			}
}

void MaxPoolLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput)
{

}

void MaxPoolLayer::updateParameters(float learningRate)
{

}

//const Tensor<float>& MaxPoolLayer::backwardPropagate(const Tensor<float>& dout, float learningRate)
//{
//	for (int n = 0; n < output.sX; n++)
//		for (int i = 0; i < output.sY; i++)
//			for (int j = 0; j < output.sZ; j++)
//			{
//				float maxVal = -INFINITY;
//				int maxK = 0, maxL = 0;
//
//				for (int k = 0; k < filterSize; k++)
//					for (int l = 0; l < filterSize; l++)
//					{
//						float val = input(n, i * filterSize + k, j * filterSize + l);
//						if (val > maxVal)
//						{
//							maxVal = val;
//							maxK = k;
//							maxL = l;
//						}
//					}
//
//				dinput(n, i * filterSize + maxK, j * filterSize + maxL) = dout(n, i, j);
//			}
//
//	return dinput;
//}
