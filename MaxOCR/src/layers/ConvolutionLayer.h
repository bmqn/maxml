#pragma once

#include "Layer.h"

class ConvolutionLayer : public Layer
{

public:
	ConvolutionLayer() = delete;
	ConvolutionLayer(int kernelSize, int kernelNum);

	virtual void forwardPropagate(const Tensor<float>& input, Tensor<float>& output) override;
	virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& dinput, const Tensor<float>& output, const Tensor<float>& doutput) override;
	virtual void updateParameters(float learningRate) override;

private:
	void oneChannelConvolution(const float* kernel, const float* src, int width, int height, float* dst);

private:
	Tensor<float>	kernel_;
	Tensor<float>	dkernel_;
	int				kernelSize_;
	int				kernelNum_;
};

