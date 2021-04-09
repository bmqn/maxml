#include "maths/Tensor.h"
#include "model/Sequential.h"
#include "utils/MnistLoader.h"

#include <iostream>

int main(void)
{
	srand(time(NULL));

	std::vector<std::pair<mocr::Tensor<double>, mocr::Tensor<double>>> data;

	{
		int numberOfImages, numberOfLabels, imageSize;
		unsigned char **img = read_mnist_images("res/train-images.idx3-ubyte", numberOfImages, imageSize);
		unsigned char *lab = read_mnist_labels("res/train-labels.idx1-ubyte", numberOfLabels);

		for (int i = 0; i < numberOfImages; i++)
		{
			mocr::Tensor<double> image(1, 28, 28);
			mocr::Tensor<double> label(1, 10, 1);

			for (int j = 0; j < 28 * 28; j++)
				image[j] = (double)img[i][j] / 255.0;

			label[(int)lab[i]] = 1.0;

			data.push_back({image, label});
		}

		for (int i = 0; i < numberOfImages; i++)
			delete[] img[i];

		delete[] img;
		delete[] lab;
	}

	mocr::Sequential seq(1 * 28 * 28);

	seq.addLayer(512, mocr::Activation::SIGMOID);
	seq.addLayer(128, mocr::Activation::SIGMOID);
	seq.addLayer(10, mocr::Activation::SIGMOID);

	for (int i = 0; i < 1000; i++)
	{
		int choice = rand() % data.size();

		mocr::Tensor<double> inp = mocr::resize(data[choice].first, 1, 28 * 28, 1);
		mocr::Tensor<double> exp = data[choice].second;

		seq.feedForward(inp);
		double loss = seq.feedBackward(exp);

		std::cout << "Iteration (" << i << "), Loss = " << loss << std::endl;
	}

	while (true)
	{
		int choice = rand() % data.size();

		mocr::Tensor<double> digi = data[choice].first;
		mocr::Tensor<double> inp = mocr::resize(digi, 1, 28 * 28, 1);
		mocr::Tensor<double> exp = data[choice].second;

		auto pred = seq.feedForward(inp);
		auto diff = mocr::sub(exp, pred);

		std::cout << "\x1B[2J\x1B[H";
		std::cout << digi.str() << std::endl;
		std::cout << diff.str() << std::endl;

		std::cin.get();
	}

	return 0;
}