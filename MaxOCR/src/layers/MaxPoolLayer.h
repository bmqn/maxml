#pragma once

#include "Layer.h"

class MaxPoolLayer : public Layer
{
public:
	MaxPoolLayer() = delete;
	MaxPoolLayer(int iN, int iWidth, int iHeight, int fSize);

	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) override;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) override;

private:
	float filterSize;
};

