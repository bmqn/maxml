#include "Layer.h"
#include "Activations.h"

#include "mocr/Tensor.h"

namespace mocr
{
	void FullyConLayer::forward(const DTensor& input, DTensor& output)
	{
		DTensor::matmul(Weights, input, output);
		DTensor::add(Biases, output, output);
	}

	void FullyConLayer::backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta)
	{
		DTensor::matmul(Weights, outputDelta, inputDelta, true, false);

		DTensor::matmul(outputDelta, input, DeltaWeights, false, true);
		DTensor::copy(outputDelta, DeltaBiases);
	}

	void FullyConLayer::update(double learningRate)
	{
		DTensor::zip(Weights, DeltaWeights, [&learningRate](double x, double y) {return x - y * learningRate; }, Weights);
		DTensor::zip(Biases, DeltaBiases, [&learningRate](double x, double y) {return x - y * learningRate; }, Biases);
	}

	void ActvLayer::forward(const DTensor& input, DTensor& output)
	{
		switch (Activation)
		{
		case ActivationFunc::SIGMOID:
			DTensor::map(input, [](double x) { return sig(x); }, output);
			break;
		case ActivationFunc::TANH:
			DTensor::map(input, [](double x) { return tanh(x); }, output);
			break;
		case ActivationFunc::RELU:
			DTensor::map(input, [](double x) { return relu(x); }, output);
			break;
		}
	}

	void ActvLayer::backward(const DTensor& input, const DTensor& output, DTensor& inputDelta, const DTensor& outputDelta)
	{
		switch (Activation)
		{
		case ActivationFunc::SIGMOID:
			DTensor::zip(output, outputDelta, [](double x, double y) { return (x * (1.0 - x)) * y; }, inputDelta);
			break;
		case ActivationFunc::TANH:
			DTensor::zip(input, outputDelta, [](double x, double y) { return (tanhPrime(x)) * y; }, inputDelta);
			break;
		case ActivationFunc::RELU:
			DTensor::zip(input, outputDelta, [](double x, double y) { return (reluPrime(x)) * y; }, inputDelta);
			break;
		}
	}
}