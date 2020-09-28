#pragma once

#include "maths/Tensor.h"

template <typename T>
class Layer
{
public:
	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) = 0;
	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) = 0;
	virtual void updateParameters(T learningRate) = 0;
};

