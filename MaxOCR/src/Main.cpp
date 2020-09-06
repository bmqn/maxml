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

int main()
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, 1);

	generator.seed(time(NULL));

	int number_of_images;
	int number_of_labels;
	int image_size;
	unsigned char** mnist_images;
	unsigned char* mnist_labels;

	mnist_images = read_mnist_images("assets/train-images.idx3-ubyte", number_of_images, image_size);
	mnist_labels = read_mnist_labels("assets/train-labels.idx1-ubyte", number_of_labels);

	// Tensor<float> t1(1, 28, 28);
	// Tensor<float> t2(1, 28, 28);

	Tensor<float> t1 = createInput(0, mnist_images);
	Tensor<float> t2 = createInput(100, mnist_images);

	/*for (int c = 0; c < t1.c_; c++)
		for (int w = 0; w < t1.w_; w++)
			for (int h = 0; h < t1.h_; h++)
				t1(c, w, h) = distribution(generator);*/

	/*for (int c = 0; c < t2.c_; c++)
		for (int w = 0; w < t2.w_; w++)
			for (int h = 0; h < t2.h_; h++)
				t2(c, w, h) = distribution(generator);*/

	Network network;

	network.setInputLayer(1, 28, 28);

	network.addConvLayer(5, 8);				// 1x28x28 -> 8x24x24
	network.addReluLayer();
	network.addMaxPoolLayer(2);				// 8x24x24 -> 8x12x12

	network.addConvLayer(3, 16);			// 8x12x12 -> 16x10x10
	network.addReluLayer();
	network.addMaxPoolLayer(2);				// 16x10x10 -> 16x5x5

	network.addFullyConnectedLayer(100);	// 16x5x5 -> 100x1x1
	network.addReluLayer();

	network.addFullyConnectedLayer(10);		// 100x1x1 -> 10x1x1
	network.addReluLayer();

	network.addSoftmaxLayer();

	network.forwardPropagate(t1);
	std::cout << network.getPredictions();

	network.forwardPropagate(t2);
	std::cout << network.getPredictions();

	for (int i = 0; i < number_of_images; i++)
		delete[] mnist_images[i];

	delete[] mnist_images;
	delete[] mnist_labels;

	//int number_of_images;
	//int number_of_labels;
	//int image_size;
	//unsigned char** mnist_images;
	//unsigned char* mnist_labels;
	//
	//std::vector<std::unique_ptr<Layer>> layers;
	//
	//layers.push_back(std::make_unique<ConvolutionLayer>(1, 28, 28, 5, 8));
	//layers.push_back(std::make_unique<ReluLayer>(8, 24, 24));
	//layers.push_back(std::make_unique<MaxPoolLayer>(8, 24, 24, 2));
	//layers.push_back(std::make_unique<FlattenLayer>(8, 12, 12));
	//layers.push_back(std::make_unique<FullyConnectedLayer>(8 * 12 * 12, 10));
	//layers.push_back(std::make_unique<SoftmaxLayer>(10));
	//
	//std::cout << "Training Started..." << std::endl;
	//
	//// TRAINING
	//{
	//	mnist_images = read_mnist_images("assets/train-images.idx3-ubyte", number_of_images, image_size);
	//	mnist_labels = read_mnist_labels("assets/train-labels.idx1-ubyte", number_of_labels);
	//
	//	float lr = 0.0005f;
	//
	//	Tensor<float, 10> dL_dy(10, 1, 1);
	//
	//	for (int e = 0; e < 50; e++)
	//	{
	//		float totalLoss = 0.0f;
	//		float totalCorrect = 0.0f;
	//
	//		std::random_device rd;
	//		std::mt19937 g(rd());
	//
	//		std::shuffle(&mnist_images[0], &mnist_images[number_of_images - 1], g);
	//
	//		for (int index = 0; index < number_of_images; index++)
	//		{
	//			// INPUT IMAGE
	//			Tensor<float> input = createInput(index, mnist_images);
	//
	//			// EXPECTED
	//			float expected[10] { 0.0f };
	//			expected[(int)mnist_labels[index]] = 1.0f;
	//
	//			// FEED FORWARD
	//			for (int i = 0; i < layers.size(); i++)
	//			{
	//				if (i == 0)
	//					layers[i]->forwardPropagate(input);
	//				else
	//					layers[i]->forwardPropagate(layers[i - 1]->output);
	//			}
	//
	//			const Tensor<float>& output = layers[layers.size() - 1]->output;
	//
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
	//
	//			// LOSS
	//			float loss = 0.0f;
	//			for (int i = 0; i < output.sX; i++)
	//				loss -= expected[i] * log(std::max(0.00001f, output(i, 0, 0)));
	//
	//			// INITIAL GRADIENT
	//			for (int i = 0; i < dL_dy.sX; i++)
	//				dL_dy(i, 0, 0) = -expected[i] / (output(i, 0, 0) + 0.001f);
	//
	//			// BACKPROPAGATION
	//			for (int i = layers.size() - 1; i >= 0; i--)
	//			{
	//				if (i == layers.size() - 1)
	//					layers[i]->backwardPropagate(dL_dy, lr);
	//				else
	//					layers[i]->backwardPropagate(layers[i + 1]->dinput, lr);
	//			}
	//
	//			totalLoss += loss;
	//			totalCorrect += correct ? 1.0f : 0.0f;
	//
	//			if (index % 500 == 0)
	//			{
	//				std::cout
	//					<< "[Epoch: " << (e + 1) << ", It: " << index << "/" << number_of_images << "]: "
	//					<< "Loss: " << std::setprecision(5) << totalLoss / (float)index << ", "
	//					<< "Accuracy: " << std::setprecision(5) << totalCorrect / (float)index * 100 << "%"
	//					<< '\r' << std::flush;
	//			}
	//		}
	//
	//		lr = std::max(lr * 0.8f, 0.0001f);
	//
	//		std::cout << std::endl;
	//	}
	//}
	//
	//std::cout << "Training Finished..." << std::endl;

	//for (int i = 0; i < number_of_images; i++)
	//	delete[] mnist_images[i];

	//delete[] mnist_images;
	//delete[] mnist_labels;
	
	return 0;
}