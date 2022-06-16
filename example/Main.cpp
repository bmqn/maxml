#include "maxml/MmlSequential.h"
#include "maxml/MmlTensor.h"

#include "MnistLoader.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>

#include <ctime>
#include <cmath>

static void RegressionExample()
{
	srand(time(nullptr));

	auto func = [](float x) -> float
	{
		// y = 2^(sin(5x^3)) - x^2
		return std::powf(2.f, std::sinf(5.f * x * x * x)) - x * x;
	};

	float step = 0.05f;
	float lower = -1.0f;
	float upper = 1.0f;

	float supremum = -std::numeric_limits<float>::infinity();

	for (float x = lower; x <= upper; x += step)
	{
		float y = func(x);

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

	std::vector<std::pair<maxml::FTensor, maxml::FTensor>> data;

	for (float x = lower; x <= upper; x += step)
	{
		data.emplace_back(maxml::FTensor{x},
						  maxml::FTensor{func(x) / supremum});
	}

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
	seqDesc.LearningRate = 0.1;
	seqDesc.LayerDescs = {
		maxml::makeInput(1, 1, 1),
		maxml::makeFullCon(16, maxml::ActivationFunc::Tanh),
		maxml::makeFullCon(8, maxml::ActivationFunc::Tanh),
		maxml::makeFullCon(1, maxml::ActivationFunc::None)
	};

	maxml::Sequential seq(seqDesc);

	{
		static constexpr size_t kNumIterations = 100000;
		static constexpr size_t kErrHistCount = 1000;
		std::vector<float> errHist;
		errHist.reserve(kNumIterations);

		std::cout << "Training for " << kNumIterations << " iterations..." << std::endl;
		for (int itr = 0; itr < kNumIterations; ++itr)
		{
			int choice = rand() % data.size();

			const maxml::FTensor &inp = data[choice].first;
			const maxml::FTensor &exp = data[choice].second;

			seq.feedForward(inp);
			errHist.push_back(seq.feedBackward(exp));

			size_t cumErrCount = errHist.size() < kErrHistCount ? errHist.size() : kErrHistCount;
			float cumErr = 0.0;
			for (size_t count = 0; count < cumErrCount; ++count)
			{
				cumErr += errHist[errHist.size() - cumErrCount + count];
			}
			float avgErr = cumErr / static_cast<float>(cumErrCount);

			std::cout << '\r' << "Iteration (" << itr + 1 << "), Error = " << std::fixed << std::setprecision(10) << avgErr;
		}
		std::cout << std::endl;
	}

	{
		int points = 50;

		std::ostringstream ss;
		ss << std::setprecision(3) << std::fixed;

		for (int i = 0; i <= points; i++)
		{
			float x = lower + (upper - lower) * ((float)i / (float)points);

			maxml::FTensor inp = {x};
			maxml::FTensor out = seq.feedForward(inp);

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
	srand(time(nullptr));

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
	seqDesc.LearningRate = 0.1;
	seqDesc.LayerDescs = {
		maxml::makeInput(1, 28, 28),
		maxml::makeConv(16, 3, 3, maxml::ActivationFunc::Tanh),
		maxml::makePool(2, 2, maxml::PoolingFunc::Max),
		maxml::makeFlatten(),
		maxml::makeFullCon(128, maxml::ActivationFunc::Tanh),
		maxml::makeFullCon(10, maxml::ActivationFunc::Sigmoid),
	};

	maxml::Sequential seq(seqDesc);

	{
		int numTrainImages;
		int trainImageSize;
		int numTrainLabels;

		unsigned char **trainImages = read_mnist_images("../res/train-images.idx3-ubyte", numTrainImages, trainImageSize);
		unsigned char *trainLabels = read_mnist_labels("../res/train-labels.idx1-ubyte", numTrainLabels);

		int trainImageWidth = static_cast<int>(std::sqrt(static_cast<float>(trainImageSize)));

		std::vector<std::pair<maxml::FTensor, maxml::FTensor>> trainData;

		for (int c = 0; c < numTrainImages; c++)
		{
			maxml::FTensor image(1, trainImageWidth, trainImageWidth);
			maxml::FTensor label(1, 10, 1);

			for (int i = 0; i < trainImageSize; i++)
			{
				image[i] = static_cast<float>(trainImages[c][i]) / 255.f;
			}

			label[static_cast<int>(trainLabels[c])] = 1.0f;

			trainData.emplace_back(image, label);
		}

		for (int i = 0; i < numTrainImages; i++)
		{
			delete[] trainImages[i];
		}
		delete[] trainImages;
		delete[] trainLabels;

		static constexpr size_t kNumIterations = 100000;
		static constexpr size_t kErrHistCount = 1000;
		std::vector<float> errHist;
		errHist.reserve(kNumIterations);

		std::cout << "Training for " << kNumIterations << " iterations..." << std::endl;
		for (int itr = 0; itr < kNumIterations; ++itr)
		{
			int choice = rand() % numTrainImages;

			seq.feedForward(trainData[choice].first);
			errHist.push_back(seq.feedBackward(trainData[choice].second));

			size_t cumErrCount = errHist.size() < kErrHistCount ? errHist.size() : kErrHistCount;
			float cumErr = 0.0;
			for (size_t count = 0; count < cumErrCount; ++count)
			{
				cumErr += errHist[errHist.size() - cumErrCount + count];
			}
			float avgErr = cumErr / static_cast<float>(cumErrCount);

			std::cout << '\r' << "Iteration (" << itr + 1 << "), Error = " << std::fixed << std::setprecision(10) << avgErr;
		}
		std::cout << std::endl;
	}

	{
		int numTestImages;
		int testImageSize;
		int numTestLabels;

		unsigned char **testImages = read_mnist_images("../res/t10k-images.idx3-ubyte", numTestImages, testImageSize);
		unsigned char *testLabels = read_mnist_labels("../res/t10k-labels.idx1-ubyte", numTestLabels);

		int testImageWidth = static_cast<int>(std::sqrt(static_cast<float>(testImageSize)));

		std::vector<std::pair<maxml::FTensor, maxml::FTensor>> testData;

		for (int c = 0; c < numTestImages; c++)
		{
			maxml::FTensor image(1, testImageWidth, testImageWidth);
			maxml::FTensor label(1, 10, 1);

			for (int i = 0; i < testImageSize; i++)
			{
				image[i] = static_cast<float>(testImages[c][i]) / 255.f;
			}

			label[static_cast<int>(testLabels[c])] = 1.0;

			testData.emplace_back(image, label);
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

			const maxml::FTensor &inp = testData[choice].first;
			const maxml::FTensor &exp = testData[choice].second;
			const maxml::FTensor &out = seq.feedForward(inp);

			float currentMax = -std::numeric_limits<float>::infinity();
			int expected = 0;

			for (int j = 0; j < exp.size(); j++)
			{
				if (exp[j] > currentMax)
				{
					currentMax = exp[j];
					expected = j;
				}
			}

			currentMax = -std::numeric_limits<float>::infinity();
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

		std::cout << numCorrect << "/" << numTests << " correct guesses, thats " << ((float)numCorrect / (float)numTests) * 100. << "%" << std::endl;
	}
}

int main(void)
{
	RegressionExample();
	// MnistExample();

	return 0;
}