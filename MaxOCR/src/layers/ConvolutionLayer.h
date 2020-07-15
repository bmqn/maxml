#pragma once

#include "Layer.h"

class ConvolutionLayer : public Layer
{

public:
	ConvolutionLayer() = delete;
	ConvolutionLayer(int iN, int iWidth, int iHeight, int kSize, int kNum);

	virtual const Tensor<float>& forwardPropagate(const Tensor<float>& input) override;
	virtual const Tensor<float>& backwardPropagate(const Tensor<float>& dout, float learningRate) override;


private:
	Tensor<float> kernel;
};

