#pragma once

#include "Layer.h"

class ReluLayer : public Layer
{

public:
	ReluLayer() = delete;
	ReluLayer(int iN, int iWidth, int iHeight);

	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) override;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) override;
};

