#pragma once

#include "Tensor.h"

class SoftmaxLayer
{

public:
	SoftmaxLayer() = delete;
	SoftmaxLayer(int iSize);

	const Tensor<float>& forwardPropagate(const Tensor<float>& input);
	const Tensor<float>& backwardPropagate(const Tensor<float>& dout);

private:	
	Tensor<float> input;
	Tensor<float> output;

	Tensor<float> dinput;
};

