#include "SoftmaxLayer.h"

template<typename T>
void SoftmaxLayer<T>::forwardPropagate(const Tensor<T>& input, Tensor<T>& output)
{
	T max = input(0, 0, 0);
	T denom = 0.0f;

	for (int i = 0; i < input.c_; i++)
		if (input(i, 0, 0) > max)
			max = input(i, 0, 0);
		
	for (int i = 0; i < input.c_; i++)
		denom += exp(input(i, 0, 0) - max);

	for (int i = 0; i < output.c_; i++)
		output(i, 0, 0) = exp(input(i, 0, 0) - max) / denom;
}

template<typename T>
void SoftmaxLayer<T>::backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput)
{
	dinput.setTo(0.0f);

	for (int p = 0; p < input.c_; p++)
		for (int i = 0; i < output.c_; i++)
		{
			if (p == i)
				dinput(p, 0, 0) += doutput(i, 0, 0) * output(p, 0, 0) * (1 - output(p, 0, 0));
			else
				dinput(p, 0, 0) += doutput(i, 0, 0) * -output(i, 0, 0) * output(p, 0, 0);
		}
}
