#include "InputLayer.h"


void InputLayer::forwardPropagate(const Tensor<float>& inputs, Tensor<float>& outputs)
{
	for (int s = 0; s < inputs.size_; s++)
		outputs[s] = inputs[s];
}