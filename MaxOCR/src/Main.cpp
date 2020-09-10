#include "utils/MnistLoader.h"
#include "utils/Tensor.h"

#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"
#include "layers/ReluLayer.h"

#include "model/Model.h"

#include <random>

static std::vector<std::pair<double, double>> data;
static int index = 0;

static void inputCallback(Tensor<double>& input)
{
	auto pair = data[index % data.size()];

	input[0] = pair.first;
}

static void expecCallback(Tensor<double>& expec)
{
	auto pair = data[index % data.size()];

	expec[0] = pair.second;
}

int main()
{
	// y = x * x + 1
	for (int i = 0; i < 1000; i++)
	{
		double x = 5.0 * (((double)i - 500.0f) / 500.0f);
		data.push_back({ x, x * x + 1.0f});
	}

	auto network = Model<double>::make(1, 1, 1, 0.00001f)
		.addFullyConnectedLayer(100)
		.addReluLayer()
		.addFullyConnectedLayer(50)
		.addReluLayer()
		.addFullyConnectedLayer(1)
		.build();

	network.setDataCallbacks(inputCallback, expecCallback);

	for (int i = 0; i < 50000; i++)
	{
		std::shuffle(data.begin(), data.end(), std::default_random_engine());
		index++;

		network.train();
	}

	Tensor<double> inp(1, 1, 1);
	Tensor<double> exp(1, 1, 1);

	for (float x = -5.0f; x <= 5.0f; x += 0.25f)
	{
		inp[0] = x;
		exp[0] = x * x + 1;
		// std::cout << "x = " << x << ", x * x = " << network.predict(inp, exp) << std::endl;

		std::cout << std::setprecision(2) << "(" << x << ", " << std::setprecision(2) << network.predict(inp, exp) << "), ";
	}

	return 0;
}