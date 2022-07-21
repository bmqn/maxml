#include "maxml/MmlSequential.h"
#include "maxml/MmlTensor.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <string>
#include <fstream>
#include <ctime>
#include <cmath>

static unsigned char **read_mnist_images(std::string full_path, int &number_of_images, int &image_size)
{
	auto reverseInt = [](int i)
	{
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	std::ifstream file(full_path, std::ios::binary);

	if (file.is_open())
	{
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051)
			throw std::runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		uchar **_dataset = new uchar *[number_of_images];
		for (int i = 0; i < number_of_images; i++)
		{
			_dataset[i] = new uchar[image_size];
			file.read((char *)_dataset[i], image_size);
		}
		return _dataset;
	}
	else
	{
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}
}

static unsigned char *read_mnist_labels(std::string full_path, int &number_of_labels)
{
	auto reverseInt = [](int i)
	{
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	std::ifstream file(full_path, std::ios::binary);

	if (file.is_open())
	{
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049)
			throw std::runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar *_dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++)
		{
			file.read((char *)&_dataset[i], 1);
		}
		return _dataset;
	}
	else
	{
		throw std::runtime_error("Unable to open file `" + full_path + "`!");
	}
}

static void RegressionExample()
{
	srand(static_cast<unsigned int>(time(nullptr)));

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

	std::vector<std::pair<maxml::Tensor, maxml::Tensor>> data;

	for (float x = lower; x <= upper; x += step)
	{
		data.emplace_back(maxml::Tensor{x},
						  maxml::Tensor{func(x) / supremum});
	}

	maxml::SequentialDesc seqDesc;
	seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
	seqDesc.LearningRate = 0.01f;
	seqDesc.LayerDescs = {
		maxml::makeInput(1, 1, 1),
		maxml::makeFullCon(16, maxml::ActivationFunc::ReLU),
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

			const maxml::Tensor &inp = data[choice].first;
			const maxml::Tensor &exp = data[choice].second;

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
	seqDesc.LearningRate = 0.001f;
	seqDesc.LayerDescs = {
		maxml::makeInput(1, 28, 28),
		maxml::makeConv(16, 5, 5, maxml::ActivationFunc::ReLU),
		maxml::makeFlatten(),
		maxml::makeFullCon(128, maxml::ActivationFunc::ReLU),
		maxml::makeFullCon(64, maxml::ActivationFunc::ReLU),
		maxml::makeFullCon(10, maxml::ActivationFunc::Softmax)
	};

	maxml::Sequential seq(seqDesc);

	{
		int numTrainImages;
		int trainImageSize;
		int numTrainLabels;

		unsigned char **trainImages = read_mnist_images("train-images.idx3-ubyte", numTrainImages, trainImageSize);
		unsigned char *trainLabels = read_mnist_labels("train-labels.idx1-ubyte", numTrainLabels);

		int trainImageWidth = static_cast<int>(std::sqrt(static_cast<float>(trainImageSize)));

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

		static constexpr size_t kNumIterations = 10000;
		static constexpr size_t kErrHistCount = 1000;
		std::vector<float> errHist;
		errHist.reserve(kNumIterations);

		std::cout << "Training for " << kNumIterations << " iterations..." << std::endl;
		for (int itr = 0; itr < kNumIterations; ++itr)
		{
			int choice = rand() % numTrainImages;

			auto out = seq.feedForward(trainData[choice].first);
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

			const maxml::Tensor &inp = testData[choice].first;
			const maxml::Tensor &exp = testData[choice].second;
			const maxml::Tensor &out = seq.feedForward(inp);

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