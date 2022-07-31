#include "maxml/MmlTensor.h"
#include "maxml/MmlSequential.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <string>
#include <fstream>
#include <ctime>
#include <cstdint>
#include <cmath>
#include <bit>

static uint8_t **ReadMnistImagesFile(std::string path, int32_t &numImages, int32_t &imageWidth, int32_t &imageHeight)
{
	auto ByteSwap = [](int32_t num)
	{
		int32_t result = (num & 0x0000FFFF) << 16 | (num & 0xFFFF0000) >> 16;
		result = (result & 0x00FF00FF) << 8 | (result & 0xFF00FF00) >> 8;
		return result;
	};

	std::ifstream f(path, std::ios::binary);
	if (f.is_open())
	{
		int32_t magicNum = 0;
		f.read((char *)&magicNum, sizeof(magicNum));
		if constexpr (std::endian::native == std::endian::little)
		{
			magicNum = ByteSwap(magicNum);
		}
		if (magicNum != 2051)
		{
			return nullptr;
		}

		f.read((char *)&numImages, sizeof(numImages));
		f.read((char *)&imageWidth, sizeof(imageWidth));
		f.read((char *)&imageHeight, sizeof(imageHeight));
		if constexpr (std::endian::native == std::endian::little)
		{
			numImages = ByteSwap(numImages);
			imageWidth = ByteSwap(imageWidth);
			imageHeight = ByteSwap(imageHeight);
		}

		uint8_t **imageData = new uint8_t*[numImages];
		for (int i = 0; i < numImages; ++i)
		{
			imageData[i] = new uint8_t[imageWidth * imageHeight];
			f.read((char*)imageData[i], imageWidth * imageHeight);
		}
		return imageData;
	}
	else
	{
		return nullptr;
	}
}

static uint8_t *ReadMnistLabelsFile(std::string path, int32_t& numLabels)
{
	auto ByteSwap = [](int32_t num)
	{
		int32_t result = (num & 0x0000FFFF) << 16 | (num & 0xFFFF0000) >> 16;
		result = (result & 0x00FF00FF) << 8 | (result & 0xFF00FF00) >> 8;
		return result;
	};

	std::ifstream f(path, std::ios::binary);
	if (f.is_open())
	{
		int32_t magicNum = 0;
		f.read((char*)&magicNum, sizeof(magicNum));
		if constexpr (std::endian::native == std::endian::little)
		{
			magicNum = ByteSwap(magicNum);
		}
		if (magicNum != 2049)
		{
			return nullptr;
		}

		f.read((char*)&numLabels, sizeof(numLabels));
		if constexpr (std::endian::native == std::endian::little)
		{
			numLabels = ByteSwap(numLabels);
		}

		uint8_t *labelData = new uint8_t[numLabels];
		for (int i = 0; i < numLabels; ++i)
		{
			f.read((char*)&labelData[i], 1);
		}
		return labelData;
	}
	else
	{
		return nullptr;
	}
}

static void RegressionExample()
{
	srand(static_cast<unsigned int>(time(nullptr)));

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
	seqDesc.LearningRate = 0.01f;
	seqDesc.LayerDescs = {
		maxml::makeInput(1, 1, 1),
		maxml::makeFullyConnected(16, maxml::ActivationFunc::ReLU),
		maxml::makeFullyConnected(16, maxml::ActivationFunc::ReLU),
		maxml::makeFullyConnected(16, maxml::ActivationFunc::ReLU),
		maxml::makeFullyConnected(1, maxml::ActivationFunc::None)
	};
	maxml::Sequential seq(seqDesc);

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

	{
		std::vector<std::pair<maxml::Tensor, maxml::Tensor>> trainData;
		for (float x = lower; x <= upper; x += step)
		{
			trainData.emplace_back(maxml::Tensor{x},
							maxml::Tensor{func(x) / supremum});
		}

		static constexpr size_t kNumIterations = 100000;
		static constexpr size_t kErrHistCount = 1000;
		std::vector<float> errHist;
		errHist.reserve(kNumIterations);

		std::cout << "Training for " << kNumIterations << " iterations..." << std::endl;
		for (int itr = 0; itr < kNumIterations; ++itr)
		{
			int choice = rand() % trainData.size();

			const auto &out = seq.feedForward(trainData[choice].first);
			auto err = seq.feedBackward(trainData[choice].second);

			errHist.push_back(err);
			float cumErr = 0.0;
			size_t cumErrCount = errHist.size() < kErrHistCount
				? errHist.size()
				: kErrHistCount;
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

			maxml::Tensor inp = {x};
			maxml::Tensor out = seq.feedForward(inp);

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
	srand(static_cast<unsigned int>(time(nullptr)));

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::CrossEntropy;
	seqDesc.LearningRate = 0.0005f;
	seqDesc.LayerDescs = {
		maxml::makeInput(1, 28, 28),
		maxml::makeConvolutional(32, 5, 5, maxml::ActivationFunc::ReLU),
		maxml::makePooling(2, 2, maxml::PoolingFunc::Max),
		maxml::makeConvolutional(32, 3, 3, maxml::ActivationFunc::ReLU),
		maxml::makePooling(2, 2, maxml::PoolingFunc::Max),
		maxml::makeFlatten(),
		maxml::makeFullyConnected(64, maxml::ActivationFunc::ReLU),
		maxml::makeFullyConnected(64, maxml::ActivationFunc::ReLU),
		maxml::makeFullyConnected(10, maxml::ActivationFunc::Softmax)
	};
	maxml::Sequential seq(seqDesc);

	{
		int numTrainImages;
		int trainImageWidth;
		int trainImageHeight;
		int numTrainLabels;

		unsigned char **trainImages = ReadMnistImagesFile("train-images.idx3-ubyte", numTrainImages, trainImageWidth, trainImageHeight);
		unsigned char *trainLabels = ReadMnistLabelsFile("train-labels.idx1-ubyte", numTrainLabels);

		if (trainImages == nullptr || trainLabels == nullptr)
		{
			std::cout << "Could not load training MNIST database!" << std::endl;
			return;
		}

		int trainImageSize = trainImageWidth * trainImageHeight;

		std::vector<std::pair<maxml::Tensor, maxml::Tensor>> trainData;

		for (int c = 0; c < numTrainImages; c++)
		{
			maxml::Tensor image(1, trainImageWidth, trainImageWidth);
			maxml::Tensor label(1, 10, 1);

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

		static constexpr size_t kNumIterations = 500000;
		static constexpr size_t kErrHistCount = 1000;
		std::vector<float> errHist;
		errHist.reserve(kNumIterations);

		std::cout << "Training for " << kNumIterations << " iterations..." << std::endl;
		for (int itr = 0; itr < kNumIterations; ++itr)
		{
			int choice = rand() % numTrainImages;

			const auto &out = seq.feedForward(trainData[choice].first);
			auto err = seq.feedBackward(trainData[choice].second);

			errHist.push_back(err);
			float cumErr = 0.0;
			size_t cumErrCount = errHist.size() < kErrHistCount
				? errHist.size()
				: kErrHistCount;
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
		int testImageWidth;
		int testImageHeight;
		int numTestLabels;

		unsigned char **testImages = ReadMnistImagesFile("../res/t10k-images.idx3-ubyte", numTestImages, testImageWidth, testImageHeight);
		unsigned char *testLabels = ReadMnistLabelsFile("../res/t10k-labels.idx1-ubyte", numTestLabels);

		if (testImages == nullptr || testLabels == nullptr)
		{
			std::cout << "Could not load testing MNIST database!" << std::endl;
			return;
		}

		int testImageSize = testImageWidth * testImageHeight;

		std::vector<std::pair<maxml::Tensor, maxml::Tensor>> testData;

		for (int c = 0; c < numTestImages; c++)
		{
			maxml::Tensor image(1, testImageWidth, testImageWidth);
			maxml::Tensor label(1, 10, 1);

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

			const auto &inp = testData[choice].first;
			const auto &exp = testData[choice].second;
			const auto &out = seq.feedForward(inp);

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
	// RegressionExample();
	MnistExample();

	return 0;
}