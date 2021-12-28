#include "mocr/Sequential.h"
#include "mocr/Tensor.h"

#include "MnistLoader.h"

#include <iostream>
#include <cmath>
#include <limits>

static void RegressionExample()
{
	auto func = [](double x) -> double {
		// y = 2x^3 - x
		return 2 * x * x * x - x;
	};

	double step = 0.05;
	double lower = -1.0;
	double upper = 1.0;
	double supremum = -std::numeric_limits<double>::infinity();

	for (double x = lower; x <= upper; x += step)
	{
		double y = func(x);

		if (y <= 0)
		{
			if (-y > supremum)
			{
				supremum = -y;
			}
		}
		else
		{
			if (y > supremum)
			{
				supremum = y;
			}
		}
	}

	std::vector<std::pair<mocr::DTensor, mocr::DTensor>> data;

	for (double x = lower; x <= upper; x += step)
	{
		data.push_back({ {x}, {func(x) / supremum} });
	}

	mocr::InputLayerDesc inpLayerDesc;
	inpLayerDesc.Channels = 1;
	inpLayerDesc.Rows = 1;
	inpLayerDesc.Cols = 1;

	mocr::FullyConnectedLayerDesc fc1LayerDesc;
	fc1LayerDesc.NumOutputs = 32;
	fc1LayerDesc.ActivFunc = mocr::ActivationFunc::Tanh;

	mocr::FullyConnectedLayerDesc fc2LayerDesc;
	fc2LayerDesc.NumOutputs = 16;
	fc2LayerDesc.ActivFunc = mocr::ActivationFunc::Tanh;

	mocr::FullyConnectedLayerDesc fc3LayerDesc;
	fc3LayerDesc.NumOutputs = 1;
	fc3LayerDesc.ActivFunc = mocr::ActivationFunc::None;

	mocr::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = mocr::LossFunc::MSE;
	seqDesc.LearningRate = 0.01;
	seqDesc.LayerDescs = { inpLayerDesc, fc1LayerDesc , fc2LayerDesc ,fc3LayerDesc };

	mocr::Sequential seq(seqDesc);

	{
		int numIterations = 50000;

		std::cout << "Training for " << numIterations << " iterations..." << std::endl;

		for (int i = 0; i < numIterations; i++)
		{
			int choice = rand() % data.size();

			const mocr::DTensor& inp = data[choice].first;
			const mocr::DTensor& exp = data[choice].second;

			const mocr::DTensor& out = seq.feedForward(inp);
			double err = seq.feedBackward(exp);

			std::cout << '\r' << "Iteration (" << i + 1 << "), Error = " << std::fixed << std::setprecision(10) << err;
		}

		std::cout << std::endl;
	}

	{
		int points = 50;

		std::ostringstream ss;
		ss << std::setprecision(3) << std::fixed;

		for (int i = 0; i <= points; i++)
		{
			double x = lower + (upper - lower) * ((double)i / (double)points);

			mocr::DTensor inp = { x };
			mocr::DTensor out = seq.feedForward(inp);

			if (i < points)
				ss << "(" << inp[0] << ", " << out[0] * supremum << "),";
			else
				ss << "(" << inp[0] << ", " << out[0] * supremum << ")" << std::endl;
		}

		std::cout << ss.str();
	}
}

static void MnistExample()
{
	mocr::InputLayerDesc inpLayerDesc;
	inpLayerDesc.Channels = 1;
	inpLayerDesc.Rows = 784;
	inpLayerDesc.Cols = 1;

	mocr::FullyConnectedLayerDesc fc1LayerDesc;
	fc1LayerDesc.NumOutputs = 256;
	fc1LayerDesc.ActivFunc = mocr::ActivationFunc::Sigmoid;

	mocr::FullyConnectedLayerDesc fc2LayerDesc;
	fc2LayerDesc.NumOutputs = 64;
	fc2LayerDesc.ActivFunc = mocr::ActivationFunc::Sigmoid;

	mocr::FullyConnectedLayerDesc fc3LayerDesc;
	fc3LayerDesc.NumOutputs = 10;
	fc3LayerDesc.ActivFunc = mocr::ActivationFunc::Sigmoid;

	mocr::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = mocr::LossFunc::MSE;
	seqDesc.LearningRate = 0.1;
	seqDesc.LayerDescs = { inpLayerDesc, fc1LayerDesc , fc2LayerDesc, fc3LayerDesc };

	mocr::Sequential seq(seqDesc);

	{
		int numTrainImages;
		int trainImageSize;
		int numTrainLabels;

		unsigned char** trainImages = read_mnist_images("../res/train-images.idx3-ubyte", numTrainImages, trainImageSize);
		unsigned char* trainLabels = read_mnist_labels("../res/train-labels.idx1-ubyte", numTrainLabels);

		int imageWidth = static_cast<int>(std::sqrt(trainImageSize));

		std::pair<mocr::DTensor, mocr::DTensor> trainData;
		trainData.first.resize(numTrainImages, trainImageSize, 1);
		trainData.second.resize(numTrainImages, 10, 1);

		for (int c = 0; c < numTrainImages; c++)
		{
			for (int i = 0; i < trainImageSize; i++)
			{
				trainData.first(c, i, 0) = trainImages[c][i];
			}

			trainData.second(c, static_cast<int>(trainLabels[c]), 0) = 1.0;
		}

		for (int i = 0; i < numTrainImages; i++)
		{
			delete[] trainImages[i];
		}
		delete[] trainImages;
		delete[] trainLabels;

		int numIterations = 50000;

		std::cout << "Training for " << numIterations  << " iterations..." << std::endl;

		for (int i = 0; i < numIterations; i++)
		{
			int choice = rand() % numTrainImages;

			const mocr::DTensor& inp = trainData.first(choice);
			const mocr::DTensor& exp = trainData.second(choice);

			const mocr::DTensor& out = seq.feedForward(inp);
			double err = seq.feedBackward(exp);

			std::cout << '\r' << "Iteration (" << i + 1 << "), Error = " << std::fixed << std::setprecision(10) << err;
		}

		std::cout << std::endl;
	}

	{
		int numTestImages;
		int testImageSize;
		int numTestLabels;

		unsigned char** testImages = read_mnist_images("../res/t10k-images.idx3-ubyte", numTestImages, testImageSize);
		unsigned char* testLabels = read_mnist_labels("../res/t10k-labels.idx1-ubyte", numTestLabels);

		int imageWidth = static_cast<int>(std::sqrt(testImageSize));

		std::pair<mocr::DTensor, mocr::DTensor> trainData;
		trainData.first.resize(numTestImages, testImageSize, 1);
		trainData.second.resize(numTestImages, 10, 1);

		for (int c = 0; c < numTestImages; c++)
		{
			for (int i = 0; i < testImageSize; i++)
			{
				trainData.first(c, i, 0) = testImages[c][i];
			}

			trainData.second(c, static_cast<int>(testLabels[c]), 0) = 1.0;
		}

		for (int i = 0; i < numTestImages; i++)
		{
			delete[] testImages[i];
		}
		delete[] testImages;
		delete[] testLabels;

		std::cout << "Testing..." << std::endl;

		int numTests = numTestImages;
		int numCorrect = 0;

		for (int i = 0; i <= numTests; i++)
		{
			int choice = rand() % numTestImages;

			const mocr::DTensor& inp = trainData.first(choice);
			const mocr::DTensor& exp = trainData.second(choice);
			const mocr::DTensor& out = seq.feedForward(inp);

			double currentMax = -std::numeric_limits<double>::infinity();
			int expected = 0;

			for (int j = 0; j < exp.size(); j++)
			{
				if (exp[j] > currentMax)
				{
					currentMax = exp[j];
					expected = j;
				}
			}

			currentMax = -std::numeric_limits<double>::infinity();
			int predicted = 0;

			for (int j = 0; j < out.size(); j++)
			{
				if (out[j] > currentMax)
				{
					currentMax = out[j];
					predicted = j;
				}
			}

			numCorrect += (expected == predicted);
		}

		std::cout << numCorrect << "/" << numTests << " correct guesses, thats " << ((double)numCorrect / (double)numTests) * 100. << "%" << std::endl;
	}
}

int main(void)
{
	RegressionExample();
	// MnistExample();

	return 0;
}