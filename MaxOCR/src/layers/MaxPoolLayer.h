#pragma once

#include "Layer.h"

class MaxPoolLayer : public Layer
{
public:
	MaxPoolLayer() = delete;
	MaxPoolLayer(int stride);

	virtual void forwardPropagate(const Tensor<float>& input, Tensor<float>& output) override;
	virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput) override;
	virtual void updateParameters(float learningRate) override;

private:
	int stride_;
};

