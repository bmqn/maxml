#pragma once

#include "Layer.h"

template<typename T>
class SoftmaxLayer : public Layer<T>
{

public:
	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override
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
	
	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override 
	{
		for (int i = 0; i < input.c_; i++)
			for (int j = 0; j < output.c_; j++)
			{
				if (i == j)
					dinput(i, 0, 0) += doutput(j, 0, 0) * output(j, 0, 0) * (1 - output(j, 0, 0));
				else
					dinput(i, 0, 0) -= doutput(j, 0, 0) * output(j, 0, 0) * output(i, 0, 0);
			}

		/*std::cout << "input" << std::endl << input << std::endl;
		std::cout << "dinput" << std::endl << dinput << std::endl;
		std::cout << "output" << std::endl << output << std::endl;
		std::cout << "doutput" << std::endl << doutput << std::endl;*/
	}

	virtual void updateParameters(T learningRate) override {}
};

