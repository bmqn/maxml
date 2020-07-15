#pragma once

#include "Layer.h"

class SoftmaxLayer : public Layer
{

public:
	SoftmaxLayer() = delete;
	SoftmaxLayer(int size);

	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) override;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) override;
};

