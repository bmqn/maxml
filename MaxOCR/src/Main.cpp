#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <random>
#include <time.h>
#include <algorithm>

#include "Tensor.h"

#include "layers/ConvolutionLayer.h"
#include "layers/MaxPoolLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"
#include "layers/ReluLayer.h"
#include "layers/FlattenLayer.h"

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

//Tensor<float> createInput(int index, unsigned char** mnist)
//{
//	Tensor<float> input(1, 28, 28);
//
//	for (int i = 0; i < 28; i++)
//		for (int j = 0; j < 28; j++)
//			input(0, i, j) = mnist[index][j + i * 28] / 255.0f;
//
//	return input;
//}

int main()
{

	Tensor<float, 3, 3> t1; // { {0, 0, 0}, { 1, 1, 1 }, { 2, 2, 2 } };
	
	std::cout << t1 << std::endl << std::endl;
	
	//for (int i = 0; i < 3; i++)
	//	for (int j = 0; j < 3; j++)
	//		std::cout << "(" << i << ", " << j << "): " << t1( i, j ) << std::endl;

	/*std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;*/

	//int number_of_images;
	//int number_of_labels;
	//int image_size;
	//unsigned char** mnist_images;
	//unsigned char* mnist_labels;

	//std::vector<std::unique_ptr<Layer>> layers;

	//layers.push_back(std::make_unique<ConvolutionLayer>(1, 28, 28, 5, 8));
	//layers.push_back(std::make_unique<ReluLayer>(8, 24, 24));
	//layers.push_back(std::make_unique<MaxPoolLayer>(8, 24, 24, 2));

	////layers.push_back(std::make_unique<ConvolutionLayer>(32, 12, 12, 3, 8));
	////layers.push_back(std::make_unique<ReluLayer>(8, 10, 10));
	////layers.push_back(std::make_unique<MaxPoolLayer>(8, 10, 10, 2));

	//layers.push_back(std::make_unique<FlattenLayer>(8, 12, 12));

	//layers.push_back(std::make_unique<FullyConnectedLayer>(8 * 12 * 12, 10));
	//layers.push_back(std::make_unique<SoftmaxLayer>(10));

	//std::cout << "Training Started..." << std::endl;

	//// TRAINING
	//{
	//	mnist_images = read_mnist_images("assets/train-images.idx3-ubyte", number_of_images, image_size);
	//	mnist_labels = read_mnist_labels("assets/train-labels.idx1-ubyte", number_of_labels);

	//	float lr = 0.0005f;

	//	Tensor<float> dL_dy(10, 1, 1);

	//	for (int e = 0; e < 50; e++)
	//	{
	//		float totalLoss = 0.0f;
	//		float totalCorrect = 0.0f;

	//		std::random_device rd;
	//		std::mt19937 g(rd());

	//		std::shuffle(&mnist_images[0], &mnist_images[number_of_images - 1], g);

	//		for (int index = 0; index < number_of_images; index++)
	//		{
	//			// INPUT IMAGE
	//			Tensor<float> input = createInput(index, mnist_images);

	//			// EXPECTED
	//			float expected[10] { 0.0f };
	//			expected[(int)mnist_labels[index]] = 1.0f;

	//			// FEED FORWARD
	//			for (int i = 0; i < layers.size(); i++)
	//			{
	//				if (i == 0)
	//					layers[i]->forwardPropagate(input);
	//				else
	//					layers[i]->forwardPropagate(layers[i - 1]->output);
	//			}

	//			const Tensor<float>& output = layers[layers.size() - 1]->output;

	//			// PREDICTION
	//			int prediction = 0;
	//			float max = -INFINITY;
	//			for (int i = 0; i < output.sX; i++)
	//				if (output(i, 0, 0) > max)
	//				{
	//					prediction = i;
	//					max = output(i, 0, 0);
	//				}
	//			bool correct = (int)mnist_labels[index] == prediction;

	//			// LOSS
	//			float loss = 0.0f;
	//			for (int i = 0; i < output.sX; i++)
	//				loss -= expected[i] * log(std::max(0.00001f, output(i, 0, 0)));

	//			// INITIAL GRADIENT
	//			for (int i = 0; i < dL_dy.sX; i++)
	//				dL_dy(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.001f);

	//			// BACKPROPAGATION
	//			for (int i = layers.size() - 1; i >= 0; i--)
	//			{
	//				if (i == layers.size() - 1)
	//					layers[i]->backwardPropagate(dL_dy, lr);
	//				else
	//					layers[i]->backwardPropagate(layers[i + 1]->dinput, lr);
	//			}

	//			totalLoss += loss;
	//			totalCorrect += correct ? 1.0f : 0.0f;

	//			if (index % 500 == 0)
	//			{
	//				std::cout
	//					<< "[Epoch: " << (e + 1) << ", It: " << index << "/" << number_of_images << "]: "
	//					<< "Loss: " << std::setprecision(5) << totalLoss / (float)index << ", "
	//					<< "Accuracy: " << std::setprecision(5) << totalCorrect / (float)index * 100 << "%"
	//					<< '\r' << std::flush;
	//			}
	//		}

	//		lr = std::max(lr * 0.8f, 0.0001f);

	//		std::cout << std::endl;
	//	}
	//}

	//std::cout << "Training Finished... Testing started" << std::endl;

	//for (int i = 0; i < number_of_images; i++)
	//	delete[] mnist_images[i];
	//delete[] mnist_images;

	//delete[] mnist_labels;

	//// TESTING
	//{
	//	mnist_images = read_mnist_images("assets/t10k-images.idx3-ubyte", number_of_images, image_size);
	//	mnist_labels = read_mnist_labels("assets/t10k-labels.idx1-ubyte", number_of_labels);

	//	float totalLoss = 0.0f;
	//	int totalCorrect = 0.0f;
	//}

	//for (int i = 0; i < number_of_images; i++)
	//	delete[] mnist_images[i];
	//delete[] mnist_images;

	//delete[] mnist_labels;

	//return 0;
}