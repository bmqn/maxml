#include "maths/Tensor.h"

#include <iostream>

int main(void)
{

	mocr::Tensor<double> x = {{1.0, 2.0},
							  {3.0, 4.0},
							  {5.0, 6.0}};

	mocr::Tensor<double> y = {{1.0, 0.0, 0.0},
							  {0.0, 1.0, 0.0}};

	mocr::Tensor z = mocr::matmul(x, y);

	auto w = mocr::resize(z, 3, 3, 1);

	std::cout << x.str() << std::endl;
	std::cout << y.str() << std::endl;
	std::cout << z.str() << std::endl;
	std::cout << w.str() << std::endl;

	return 0;
}

// int main()
// {
// 	// TODO: Need a flatten layer....

// 	std::vector<std::pair<Tensor<double>, Tensor<double>>> data;

// 	for (double i = -500.0; i < 500.0; i++)
// 	{
// 		double x = 1.0 * (i / 500.0);

// 		Tensor<double> inp(1, 1, 1, {x});
// 		Tensor<double> exp(1, 1, 1, {x * x + 1});

// 		data.push_back({std::move(inp), std::move(exp)});
// 	}

// 	auto network = make()
// 					   .addFullyConnectedLayer(25)
// 					   .addReluLayer()
// 					   .addFullyConnectedLayer(5)
// 					   .addReluLayer()
// 					   .addFullyConnectedLayer(1)
// 					   .build();

// 	std::default_random_engine generator;
// 	generator.seed(time(NULL));

// 	int index = 0;
// 	double learningRate = 0.01f;

// 	for (int epoch = 0; epoch < 20; epoch++)
// 	{
// 		network.beginEpoch();

// 		std::shuffle(data.begin(), data.end(), generator);

// 		for (; index < data.size(); index++)
// 			network.train(data[index].first, data[index].second, learningRate);

// 		index = 0;
// 		learningRate *= 0.9;

// 		network.endEpoch();
// 	}

// 	for (double x = -1.0; x <= 1.0; x += 0.1)
// 	{
// 		network.test(Tensor<double>(1, 1, 1, {x}));
// 	}

// 	return 0;
// }