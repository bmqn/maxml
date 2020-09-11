#include "utils/MnistLoader.h"
#include "utils/Tensor.h"

#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"
#include "layers/ReluLayer.h"

#include "model/Model.h"

#include <random>

int main()
{
	std::vector<std::pair<double, double>> data;

	for (double i = -1000.0; i < 1000.0; i++)
	{
		double x = 3.14159 * (i / 1000.0);

		data.push_back({ x, std::sin(x)});
	}

	auto network = Model<double>::make(1, 1, 1, 0.0001f)
		.addFullyConnectedLayer(100)
		.addReluLayer()
		.addFullyConnectedLayer(50)
		.addReluLayer()
		.addFullyConnectedLayer(1)
		.build();

	Tensor<double> inp(1, 1, 1);
	Tensor<double> exp(1, 1, 1);
	int index = 0;

	for (int epoch = 0; epoch < 1000; epoch++)
	{
		std::cout << "Epoch " << epoch << " starting..." << std::endl;

		std::shuffle(data.begin(), data.end(), std::default_random_engine());

		for (; index < data.size(); index++)
		{
			inp[0] = data[index].first;
			exp[0] = data[index].second;

			network.train(&inp, &exp);
		}

		index = 0;
	}

	std::cout << "Testing starting..." << std::endl;
	for (double x = -1.0; x <= 1.0; x += 0.05)
	{
		double theta = 3.14159 * x;

		inp[0] = theta;
		exp[0] = std::sin(theta);

		std::cout << std::setprecision(2) << "(" << theta << ", " << network.predict(&inp, &exp) << "), ";
	}

	return 0;
}