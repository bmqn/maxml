#pragma once

#include "../utils/Tensor.h"

class Layer
{
public:
	virtual void forwardPropagate(const Tensor<float>& input, Tensor<float>& output) = 0;
	virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput) = 0;
	virtual void updateParameters(float learningRate) = 0;
};

