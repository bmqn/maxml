#pragma once

#include "Tensor.h"

class ReluLayer
{

public:
	ReluLayer() = delete;
	ReluLayer(int iN, int iWidth, int iHeight);

	const Tensor<float>& forwardPropagate(const Tensor<float>& input);
	const Tensor<float>& backwardPropagate(const Tensor<float>& dout);

private:
	Tensor<float> input;
	Tensor<float> output;

	Tensor<float> dinput;

};

