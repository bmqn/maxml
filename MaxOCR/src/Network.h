#pragma once

#include "layers/Layer.h"
#include "utils/Tensor.h"

#include <vector>
#include <functional>
#include <utility>

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
	void addInputLayer(int channels, int width, int height);
	void addOutputLayer(int outputs);

	void addConvLayer(int kernelSize, int kernelNum);
	void addMaxPoolLayer(int stride);
	void addReluLayer();
	void addFullyConnectedLayer(int outputSize);
	void addSoftmaxLayer();

	void forwardPropagate();
	void backwardPropagate();
	void updateParameters();

	void setCallback(std::function<void(const std::pair<Tensor<float>*, Tensor<float>*>)> dataCallback)
	{
		dataCallback_ = dataCallback;
	}

private:
	const Tensor<float>* getInputData(int layer) const;
	const Tensor<float>* getOutputData(int layer) const;

	Tensor<float>* getInputData(int layer);
	Tensor<float>* getOutputData(int layer);

	const Tensor<float>* getInputGradient(int layer) const;
	const Tensor<float>* getOutputGradient(int layer) const;

	Tensor<float>* getInputGradient(int layer);
	Tensor<float>* getOutputGradient(int layer);

private:
	std::vector<std::shared_ptr<Layer>> layers_;

	Tensor<float>* input_;
	Tensor<float> expected_ = Tensor<float>(1, 1, 1);

	std::vector<Tensor<float>> data_;
	std::vector<Tensor<float>> gradient_;

	std::function<void(const std::pair<Tensor<float>*, Tensor<float>*>)> dataCallback_;
};

