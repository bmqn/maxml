#pragma once

#include "Layer.h"
#include "Tensor.h"

class FullyConnectedLayer : public Layer
{

public:
	FullyConnectedLayer() = delete;
	FullyConnectedLayer(int iSize, int oSize);

	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) override;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) override;
	
private:
	Tensor<float> weights;
	Tensor<float> biases;
};

