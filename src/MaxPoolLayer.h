#pragma once

#include "Tensor.h"

class MaxPoolLayer
{
public:
	MaxPoolLayer() = delete;
	MaxPoolLayer(int iN, int iWidth, int iHeight, int fSize);
	~MaxPoolLayer();

	const Tensor<float>& forwardPropagate(const Tensor<float>& input);
	const Tensor<float>& backwardPropagate(const Tensor<float>& dout);

private:
	float filterSize;

	Tensor<float> input;
	Tensor<float> output;

	Tensor<float > gradin;
};

