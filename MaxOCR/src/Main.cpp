
#include "Common.h"

#include "utils/MnistLoader.h"

#include "maths/Tensor.h"

#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"
#include "layers/ReluLayer.h"

#include "model/Model.h"

int main()
{
	// TODO: Need a flatten layer....

	std::vector<std::pair<Tensor<double>, Tensor<double>>> data;

	for (double i = -500.0; i < 500.0; i++)
	{
		double x = 1.0 * (i / 500.0);

		Tensor<double> inp(1, 1, 1, {x});
		Tensor<double> exp(1, 1, 1, {x * x + 1});

		data.push_back({ std::move(inp), std::move(exp) });
	}

	auto network = 
		Model::make(1, 1, 1)
		.addFullyConnectedLayer(25)
		.addReluLayer()
		.addFullyConnectedLayer(5)
		.addReluLayer()
		.addFullyConnectedLayer(1)
		.build();

	std::default_random_engine generator;
	generator.seed(time(NULL));

	int index = 0;
	double learningRate = 0.01f;

	for (int epoch = 0; epoch < 20; epoch++)
	{
		network.beginEpoch();

		std::shuffle(data.begin(), data.end(), generator);

		for (; index < data.size(); index++)
			network.train(data[index].first, data[index].second, learningRate);

		index = 0;
		learningRate *= 0.9;

		network.endEpoch();
	}

	for (double x = -1.0; x <= 1.0; x += 0.1)
	{
		network.test(Tensor<double>(1, 1, 1, { x }));
	}

	return 0;
}