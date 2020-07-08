#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <random>
#include <time.h>

#include "ConvolutionLayer.h"
#include "MaxPoolLayer.h"
#include "FullyConnectedLayer.h"
#include "SoftmaxLayer.h"
#include "ReluLayer.h"

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

static unsigned char** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
	
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	std::ifstream file(full_path, std::ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

		file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		uchar** _dataset = new uchar * [number_of_images];
		for (int i = 0; i < number_of_images; i++) {
			_dataset[i] = new uchar[image_size];
			file.read((char*)_dataset[i], image_size);
		}
		return _dataset;
	}
	else {
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}
}

static unsigned char* read_mnist_labels(std::string full_path, int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	std::ifstream file(full_path, std::ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

		file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw std::runtime_error("Unable to open file `" + full_path + "`!");
	}
}

int main()
{
	int number_of_images;
	int number_of_labels;
	int image_size;
	unsigned char** mnist_images;
	unsigned char* mnist_labels;

	ConvolutionLayer layer1(1, 28, 28, 3, 8);
	ReluLayer layer2(8, 26, 26);
	MaxPoolLayer layer3(8, 26, 26, 2);
	FullyConnectedLayer layer4(8, 13, 13, 10);
	SoftmaxLayer layer5(10);

	Tensor<float> input(1, 28, 28);
	Tensor<float> expected(10, 1, 1);

	Tensor<float> dL_dy(10, 1, 1);

	std::cout << "Training Started..." << std::endl;

	// TRAINING
	{
		mnist_images = read_mnist_images("assets/train-images.idx3-ubyte", number_of_images, image_size);
		mnist_labels = read_mnist_labels("assets/train-labels.idx1-ubyte", number_of_labels);

		int totalIterations = 0;

		for (int e = 0; e < 50; e++)
		{
			float totalLoss = 0.0f;
			int totalCorrect = 0.0f;

			for (int k = 0; k < number_of_images; k++)
			{
				int index = k;

				// INPUT IMAGE
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						input(0, i, j) = mnist_images[index][j + i * 28] / 255.0f;

				// EXPECTED
				for (int i = 0; i < 10; i++)
					expected(i, 0, 0) = 0.0f;
				expected((int)mnist_labels[index], 0, 0) = 1.0f;

				// FEED FORWARD
				const Tensor<float>& output = layer5.forwardPropagate(layer4.forwardPropagate(layer3.forwardPropagate(layer2.forwardPropagate(layer1.forwardPropagate(input)))));

				// PREDICTION
				int prediction = 0;
				float max = -INFINITY;
				for (int i = 0; i < output.sX; i++)
					if (output(i, 0, 0) > max)
					{
						prediction = i;
						max = output(i, 0, 0);
					}
				bool correct = (int)mnist_labels[index] == prediction;

				// LOSS
				float loss = 0.0f;
				for (int i = 0; i < output.sX; i++)
					loss -= expected(i, 0, 0) * log(std::max(0.00001f, output(i, 0, 0)));

				if (k % 10000 == 0) std::cout << "Loss: " << std::setprecision(5) << loss << std::endl;

				// INITIAL GRADIENT
				for (int i = 0; i < dL_dy.sX; i++)
					dL_dy(i, 0, 0) = -expected(i, 0, 0) / (output(i, 0, 0) + 0.001f);

				float lr = 0.001f / (1.0f + (float)e / 5.0f);

				// BACKPROPAGATION
				layer1.backwardPropagate(layer2.backwardPropagate(layer3.backwardPropagate(layer4.backwardPropagate(layer5.backwardPropagate(dL_dy), lr))), lr);


				totalLoss += loss;
				totalCorrect += correct ? 1 : 0;
			}

			totalIterations += number_of_images;

			float avgLoss = totalLoss / (float)number_of_images;
			float avgCorrect = (float)totalCorrect / (float)number_of_images;

			std::cout
				<< "[Epoch: " << (e + 1) << ", It: " << totalIterations << "]: "
				<< "Loss: " << std::setprecision(5) << avgLoss
				<< ", Accuracy: " << std::setprecision(5) << (avgCorrect * 100) << "%"
				<< std::endl;
		}
	}

	std::cout << "Training Finished... Testing started" << std::endl;

	for (int i = 0; i < number_of_images; i++)
		delete[] mnist_images[i];
	delete[] mnist_images;

	delete[] mnist_labels;

	//cv::namedWindow("Image", cv::WindowFlags::WINDOW_NORMAL);
	//cv::resizeWindow("Image", 600, 600);

	//for (int i = 0; i < filtered.sX; i++)
	//{
	//	cv::Mat image = cv::Mat(filtered.sY, filtered.sZ, CV_32FC1, &filtered(i, 0, 0));
	//	cv::imshow("Image", image);
	//	cv::waitKey();
	//}

	// TESTING
	{
		mnist_images = read_mnist_images("assets/t10k-images.idx3-ubyte", number_of_images, image_size);
		mnist_labels = read_mnist_labels("assets/t10k-labels.idx1-ubyte", number_of_labels);

		float totalLoss = 0.0f;
		int totalCorrect = 0.0f;

		for (int e = 0; e < number_of_images; e++)
		{
			for (int i = 0; i < 28; i++)
				for (int j = 0; j < 28; j++)
					input(0, i, j) = mnist_images[e][j + i * 28] / 255.0f;

			for (int i = 0; i < 10; i++)
				expected(i, 0, 0) = 0.0f;
			expected((int)mnist_labels[e], 0, 0) = 1.0f;

			const Tensor<float>& output = layer5.forwardPropagate(layer4.forwardPropagate(layer3.forwardPropagate(layer2.forwardPropagate(layer1.forwardPropagate(input)))));

			// PREDICTION
			int prediction = 0;
			float max = -INFINITY;
			for (int i = 0; i < output.sX; i++)
				if (output(i, 0, 0) > max)
				{
					prediction = i;
					max = output(i, 0, 0);
				}
			bool correct = (int)mnist_labels[e] == prediction;

			// LOSS
			float loss = 0.0f;
			for (int i = 0; i < output.sX; i++)
				loss -= expected(i, 0, 0) * log(std::max(0.00001f, output(i, 0, 0)));

			totalLoss += loss;
			totalCorrect += correct ? 1 : 0;
		}

		float avgLoss = totalLoss / (float)number_of_images;
		float avgCorrect = (float)totalCorrect / (float)number_of_images;

		std::cout
			<< "Testing Results: "
			<< "Loss: " << std::setprecision(5) << avgLoss
			<< ", Accuracy: " << std::setprecision(5) << (avgCorrect * 100) << "%"
			<< std::endl;
	}

	for (int i = 0; i < number_of_images; i++)
		delete[] mnist_images[i];
	delete[] mnist_images;

	delete[] mnist_labels;

	return 0;
}