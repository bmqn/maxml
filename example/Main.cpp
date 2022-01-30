#include "maxml/MmlSequential.h"
#include "maxml/MmlTensor.h"

#include "MnistLoader.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>

static void RegressionExample()
{
	/*auto func = [](double x) -> double {
		// y = abs(x)^(2cos(3x)) - 1
		return std::pow(std::abs(x), 2 * std::cos(3 * x)) - 1;
	};*/

	auto func = [](double x) -> double {
		// y = 2^(sin(5x^3)) - x^2
		return std::pow(2, std::sin(5 * x * x * x)) - x * x;
	};

	double step = 0.05;
	double lower = -1.0;
	double upper = 1.0;
	
	double supremum = -std::numeric_limits<double>::infinity();

	for (double x = lower; x <= upper; x += step)
	{
		double y = func(x);

		if (y < 0)
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

	std::vector<std::pair<maxml::DTensor, maxml::DTensor>> data;

	for (double x = lower; x <= upper; x += step)
	{
		data.push_back({ {x}, {func(x) / supremum} });
	}

	maxml::InputLayerDesc inpLayerDesc;
	inpLayerDesc.Channels = 1;
	inpLayerDesc.Rows = 1;
	inpLayerDesc.Cols = 1;

	maxml::FullyConnectedLayerDesc fc1LayerDesc;
	fc1LayerDesc.NumOutputs = 16;
	fc1LayerDesc.ActivFunc = maxml::ActivationFunc::Tanh;

	maxml::FullyConnectedLayerDesc fc2LayerDesc;
	fc2LayerDesc.NumOutputs = 8;
	fc2LayerDesc.ActivFunc = maxml::ActivationFunc::Tanh;

	maxml::FullyConnectedLayerDesc fc3LayerDesc;
	fc3LayerDesc.NumOutputs = 1;
	fc3LayerDesc.ActivFunc = maxml::ActivationFunc::None;

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
	seqDesc.LearningRate = 0.1;
	seqDesc.LayerDescs = { inpLayerDesc, fc1LayerDesc , fc2LayerDesc ,fc3LayerDesc };

	maxml::Sequential seq(seqDesc);

	{
		int numIterations = 500000;

		std::cout << "Training for " << numIterations << " iterations..." << std::endl;

		for (int i = 0; i < numIterations; i++)
		{
			int choice = rand() % data.size();

			const maxml::DTensor& inp = data[choice].first;
			const maxml::DTensor& exp = data[choice].second;

			const maxml::DTensor& out = seq.feedForward(inp);
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

			maxml::DTensor inp = { x };
			maxml::DTensor out = seq.feedForward(inp);

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
	maxml::InputLayerDesc inpLayerDesc;
	inpLayerDesc.Channels = 1;
	inpLayerDesc.Rows = 784;
	inpLayerDesc.Cols = 1;

	maxml::FullyConnectedLayerDesc fc1LayerDesc;
	fc1LayerDesc.NumOutputs = 256;
	fc1LayerDesc.ActivFunc = maxml::ActivationFunc::Sigmoid;

	maxml::FullyConnectedLayerDesc fc2LayerDesc;
	fc2LayerDesc.NumOutputs = 64;
	fc2LayerDesc.ActivFunc = maxml::ActivationFunc::Sigmoid;

	maxml::FullyConnectedLayerDesc fc3LayerDesc;
	fc3LayerDesc.NumOutputs = 10;
	fc3LayerDesc.ActivFunc = maxml::ActivationFunc::Sigmoid;

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
	seqDesc.LearningRate = 0.1;
	seqDesc.LayerDescs = { inpLayerDesc, fc1LayerDesc , fc2LayerDesc, fc3LayerDesc };

	maxml::Sequential seq(seqDesc);

	{
		int numTrainImages;
		int trainImageSize;
		int numTrainLabels;

		unsigned char** trainImages = read_mnist_images("../res/train-images.idx3-ubyte", numTrainImages, trainImageSize);
		unsigned char* trainLabels = read_mnist_labels("../res/train-labels.idx1-ubyte", numTrainLabels);

		int imageWidth = static_cast<int>(std::sqrt(trainImageSize));

		std::pair<maxml::DTensor, maxml::DTensor> trainData;
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

			const maxml::DTensor& inp = trainData.first(choice);
			const maxml::DTensor& exp = trainData.second(choice);

			const maxml::DTensor& out = seq.feedForward(inp);
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

		std::pair<maxml::DTensor, maxml::DTensor> trainData;
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

			const maxml::DTensor& inp = trainData.first(choice);
			const maxml::DTensor& exp = trainData.second(choice);
			const maxml::DTensor& out = seq.feedForward(inp);

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