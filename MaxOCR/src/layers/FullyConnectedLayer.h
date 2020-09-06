#pragma once

#include "Layer.h"

class FullyConnectedLayer : public Layer
{

public:
	FullyConnectedLayer() = delete;
	FullyConnectedLayer(int inputSize, int outputSize);

	virtual void forwardPropagate(const Tensor<float>& input, Tensor<float>& output) override;
	virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput) override;
	virtual void updateParameters(float learningRate) override;

private:
	Tensor<float>	weights_;
	Tensor<float>	biases_;

	Tensor<float>	dweights_;
	Tensor<float>	dbiases_;
	int				inputSize_;
	int				outputSize_;

};

