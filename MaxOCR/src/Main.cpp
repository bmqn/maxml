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
	int numberOfImages, numberOfLabels, imageSize;
	unsigned char** img = read_mnist_images("assets/train-images.idx3-ubyte", numberOfImages, imageSize);
	unsigned char* lab = read_mnist_labels("assets/train-labels.idx1-ubyte", numberOfLabels);

	std::vector<std::pair<Tensor<double>, Tensor<double>>> data;
	std::vector<Tensor<double>> images;
	
	for (int i = 0; i < numberOfImages; i++)
	{
		data.push_back( {Tensor<double>(1, 28, 28), Tensor<double>(10, 1, 1) });

		for (int j = 0; j < 28 * 28; j++)
			data[i].first[j] = (double) img[i][j] / 255.0 - 0.5;

		data[i].second[(int)lab[i]] = 1.0;
	}

	for (int i = 0; i < numberOfImages; i++)
		delete[] img[i];
	
	delete[] img;
	delete[] lab;

	auto network = Model<double>::make(1, 28, 28)
		.addFullyConnectedLayer(64)
		.addReluLayer()
		.addFullyConnectedLayer(10)
		.addReluLayer()
		.addSoftmaxLayer()
		.build();

	int index = 0;
	double learningRate = 0.001f;

	for (int epoch = 0; epoch < 5; epoch++)
	{
		network.beginEpoch();

		std::shuffle(data.begin(), data.end(), std::default_random_engine());

		for (; index < data.size(); index++)
		{
			network.train(data[index].first, data[index].second, learningRate);
			std::cout << index << ", Loss " << network.getLoss() << '\r';
		}

		std::cout << std::endl;

		index = 0;
		learningRate *= 0.99;

		network.endEpoch();
	}

	img = read_mnist_images("assets/t10k-images.idx3-ubyte", numberOfImages, imageSize);
	lab = read_mnist_labels("assets/t10k-labels.idx1-ubyte", numberOfLabels);

	data.clear();

	for (int i = 0; i < numberOfImages; i++)
	{
		data.push_back({ Tensor<double>(1, 28, 28), Tensor<double>(10, 1, 1) });

		for (int j = 0; j < 28 * 28; j++)
			data[i].first[j] = (double)img[i][j] / 255.0 - 0.5;

		data[i].second[(int)lab[i]] = 1.0;
	}

	for (int i = 0; i < numberOfImages; i++)
		delete[] img[i];

	delete[] img;
	delete[] lab;

	std::cout << "Testing starting..." << std::endl;
	for (int i = 0; i < numberOfImages; i++)
	{
		std::cout << data[i].second << std::endl << network.test(data[i].first) << std::endl;
		std::cout << "--------------------------------------------" << std::endl;
	}

	return 0;
}