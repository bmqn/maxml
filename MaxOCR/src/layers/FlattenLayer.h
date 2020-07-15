#pragma once

#include "Layer.h"
#include "Tensor.h"

class FlattenLayer : public Layer
{
public:
	FlattenLayer() = delete;
	FlattenLayer(int iN, int iWidth, int iHeight);

	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) override;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) override;

};

