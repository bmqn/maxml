#pragma once

#include "Tensor.h"

class FullyConnectedLayer
{

public:
	FullyConnectedLayer() = delete;
	FullyConnectedLayer(int iN, int iWidth, int iHeight, int oSize);

	const Tensor<float>& forwardPropagate(const Tensor<float>& input);
	const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate);
	
private:
	Tensor<float> input;
	Tensor<float> output;

	Tensor<float> weights;
	Tensor<float> biases;
	
	Tensor<float> dinput;
};

