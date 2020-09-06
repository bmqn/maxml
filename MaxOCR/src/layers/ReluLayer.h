#pragma once

#include "Layer.h"

class ReluLayer : public Layer
{

public:
	virtual void forwardPropagate(const Tensor<float>& input, Tensor<float>& output) override;
	virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput) override;
	virtual void updateParameters(float learningRate) override;
};

