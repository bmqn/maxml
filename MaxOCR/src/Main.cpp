#include "utils/MnistLoader.h"

#include "utils/Tensor.h"

#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"
#include "layers/ReluLayer.h"

#include "Network.h"

#include <random>

Tensor<float> createInput(int index, unsigned char** mnist)
{
	Tensor<float> input(1, 28, 28);

	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
			input(0, i, j) = mnist[index][i * 28 + j] / 255.0f;

	return input;
}

static std::vector<std::pair<float, float>> data;

static int index = 0;

static void inputCallback(Tensor<float>& input)
{
	auto pair = data[index % data.size()];

	input[0] = pair.first;
}

static void expecCallback(Tensor<float>& expec)
{
	auto pair = data[index % data.size()];

	expec[0] = pair.second;
}

int main()
{
	int number_of_images;
	int number_of_labels;
	int image_size;
	unsigned char** mnist_images;
	unsigned char* mnist_labels;

	mnist_images = read_mnist_images("assets/train-images.idx3-ubyte", number_of_images, image_size);
	mnist_labels = read_mnist_labels("assets/train-labels.idx1-ubyte", number_of_labels);

	for (int i = 0; i < 1000; i++)
	{
		float x = 3 * (((float)i - 500.0f) / 500.0f);
		data.push_back({ x, x * x});
	}

	Network network;

	network.addInputLayer(1, 1, 1);
	network.addFullyConnectedLayer(10);
	network.addReluLayer();
	network.addFullyConnectedLayer(10);
	network.addReluLayer();
	network.addFullyConnectedLayer(1);

	network.formNetwork();

	network.setDataCallbacks(inputCallback, expecCallback);

	for (int i = 0; i < 10000; i++)
	{
		std::shuffle(data.begin(), data.end(), std::default_random_engine());

		network.train();

		index++;
	}

	for (int i = 0; i < number_of_images; i++)
		delete[] mnist_images[i];

	delete[] mnist_images;
	delete[] mnist_labels;

	return 0;
}