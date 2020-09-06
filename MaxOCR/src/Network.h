#pragma once

#include "layers/Layer.h"
#include "utils/Tensor.h"

#include <vector>

namespace layer
{

struct Input
{
	int channels;
	int width;
	int height;
};

struct Convolution
{
	int kernelSize;
	int kernelNum;
};

struct MaxPool
{
	int stride;
};

}

class Network
{
public:
	void setInputLayer(int channels, int width, int height);
	void addConvLayer(int kernelSize, int kernelNum);
	void addMaxPoolLayer(int stride);
	void addReluLayer();
	void addFullyConnectedLayer(int outputSize);
	void addSoftmaxLayer();

	void forwardPropagate(const Tensor<float>& data);

	const Tensor<float> getPredictions() const;

private:
	const Tensor<float>* getInputData(int layer) const;
	const Tensor<float>* getOutputData(int layer) const;

	Tensor<float>* getInputData(int layer);
	Tensor<float>* getOutputData(int layer);

private:
	std::vector<std::shared_ptr<Layer>> layers_;
	std::vector<Tensor<float>> data_;
};

