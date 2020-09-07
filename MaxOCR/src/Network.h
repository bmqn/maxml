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

	void addConvLayer(int kernelSize, int kernelNum);
	void addMaxPoolLayer(int stride);
	void addReluLayer();
	void addFullyConnectedLayer(int outputSize);
	void addSoftmaxLayer();

	void formNetwork();

	void setDataCallbacks(std::function<void(Tensor<float>&)> inputCallback, std::function<void(Tensor<float>&)> expecCallback)
	{
		this->inputCallback_ = inputCallback;
		this->expecCallback_ = expecCallback;
	}

	void train();
	Tensor<float> predict(const Tensor<float>& input);

private:
	void forwardPropagate();
	void backwardPropagate();
	void updateParameters();


private:
	int inputSize_;
	int outputSize_;

	std::vector<std::shared_ptr<Layer>> layers_;

	std::shared_ptr<Tensor<float>> expected_;
	std::vector<Tensor<float>> data_; // Stores the input and output of each layer.
	std::vector<Tensor<float>> gradient_;

	std::function<void(Tensor<float>&)> inputCallback_;
	std::function<void(Tensor<float>&)> expecCallback_;
};

