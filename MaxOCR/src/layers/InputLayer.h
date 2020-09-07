#pragma once

#include "Layer.h"

class InputLayer : public Layer
{
	virtual void forwardPropagate(const Tensor<float>& inputs, Tensor<float>& outputs) override;
	virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput) override {}
	virtual void updateParameters(float learningRate) override {}
};

