#pragma once

#include "Layer.h"

template<typename T>
class SoftmaxLayer : public Layer<T>
{

public:
	virtual void forwardPropagate(const Tensor<T>& input, Tensor<T>& output) override;
	virtual void backwardPropagate(const Tensor<T>& input, Tensor<T>& dinput, const Tensor<T>& output, const Tensor<T>& doutput) override;
	virtual void updateParameters(T learningRate) override {}
};

