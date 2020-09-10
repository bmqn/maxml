#pragma once

#include "Layer.h"

template<typename T>
class MaxPoolLayer : public Layer<T>
{
public:
	MaxPoolLayer() = delete;
	MaxPoolLayer(int stride);

	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override;
	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override;
	virtual void updateParameters(T learningRate) override {}

private:
	int stride_;
};

