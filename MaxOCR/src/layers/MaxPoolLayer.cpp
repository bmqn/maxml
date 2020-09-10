#include "MaxPoolLayer.h"



template<typename T>
MaxPoolLayer<T>::MaxPoolLayer(int stride)
	: stride_(stride)
{
}

template<typename T>
void MaxPoolLayer<T>::forwardPropagate(const Tensor<T>& input, Tensor<T>& output)
{
	for (int c = 0; c < output.c_; c++)
		for (int w = 0; w < output.w_; w++)
			for (int h = 0; h < output.h_; h++)
			{
				T maxVal = input(c, w * stride_, h * stride_);

				for (int i = 0; i < stride_; i++)
					for (int j = 0; j < stride_; j++)
					{
						T val = input(c, w * stride_ + i, h * stride_ + j);

						if (val > maxVal)
							maxVal = val;
					}

				output(c, w, h) = maxVal;
			}
}

template<typename T>
void MaxPoolLayer<T>::backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput)
{
	for (int c = 0; c < output.c_; c++)
		for (int w = 0; w < output.w_; w++)
			for (int h = 0; h < output.h_; h++)
			{
				T maxVal = input(c, w * stride_, h * stride_);
				int maxI = 0, maxJ = 0;

				for (int i = 0; i < stride_; i++)
					for (int j = 0; j < stride_; j++)
					{
						T val = input(c, w * stride_ + i, h * stride_ + j);
						if (val > maxVal)
						{
							maxVal = val;
							maxI = i;
							maxJ = j;
						}
					}

				dinput(c, w * stride_ + maxI, h * stride_ + maxJ) += doutput(c, w, h);
			}
}
