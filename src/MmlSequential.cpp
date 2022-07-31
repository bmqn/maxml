#include "maxml/MmlSequential.h"
#include "MmlLayer.h"
#include "MmlLog.h"

namespace maxml
{
	InputDesc makeInput(size_t channels, size_t rows, size_t cols)
	{
		return { channels, rows, cols };
	}

	FullyConnectedDesc makeFullyConnected(size_t numOutputs, ActivationFunc activFunc)
	{
		return { numOutputs, activFunc };
	}

	ConvolutionalDesc makeConvolutional(size_t numKernels, size_t kernelWidth, size_t kernelHeight, ActivationFunc activFunc)
	{
		return { numKernels, kernelWidth, kernelHeight, activFunc };
	}

	PoolingDesc makePooling(size_t tileWidth, size_t tileHeight, PoolingFunc poolingFunc)
	{
		return { tileWidth, tileHeight, poolingFunc };
	}

	FlattenDesc makeFlatten()
	{
		return {};
	}

	Sequential::Sequential(const SequentialDesc &sequentialDesc)
	{
		m_SequentialDesc = sequentialDesc;
		m_ObjectiveFunc = sequentialDesc.ObjectiveFunc;
		m_LearningRate = sequentialDesc.LearningRate;

		size_t inChannels = 0;
		size_t inRows = 0;
		size_t inCols = 0;

		size_t outChannels = 0;
		size_t outRows = 0;
		size_t outCols = 0;

		auto MakeInputOutputPair = [&]() {
			m_Data.push_back(std::make_pair(
				m_Data.empty()
				? std::make_shared<Tensor>(inChannels, inRows, inCols)
				: m_Data.back().second,
				std::make_shared<Tensor>(outChannels, outRows, outCols)
			));
		};

		auto MakeDeltaInputOutputPair = [&]() {
			m_Delta.push_back(std::make_pair(
				m_Delta.empty()
				? std::make_shared<Tensor>(inChannels, inRows, inCols)
				: m_Delta.back().second,
				std::make_shared<Tensor>(outChannels, outRows, outCols)
			));
		};

		for (auto it = sequentialDesc.LayerDescs.begin(); it != sequentialDesc.LayerDescs.end(); it++)
		{
			SequentialDesc::LayerDesc layerDesc = *it;

			if (std::holds_alternative<InputDesc>(layerDesc))
			{
				MML_ASSERT(it == sequentialDesc.LayerDescs.begin(), "Cannot have more than one input layer!");

				InputDesc inpLayerDesc = std::get<InputDesc>(layerDesc);

				inChannels = inpLayerDesc.Channels;
				inRows = inpLayerDesc.Rows;
				inCols = inpLayerDesc.Cols;

				outChannels = inpLayerDesc.Channels;
				outRows = inpLayerDesc.Rows;
				outCols = inpLayerDesc.Cols;
			}
			else if (std::holds_alternative<FullyConnectedDesc>(layerDesc))
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				FullyConnectedDesc fcLayerDesc = std::get<FullyConnectedDesc>(layerDesc);
				size_t numInputs = outRows;
				size_t numOutputs = fcLayerDesc.NumOutputs;
				ActivationFunc activFunc = fcLayerDesc.ActivFunc;

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = 1;
				outRows = numOutputs;
				outCols = 1;

				// Fully connected layer
				{
					float sigma;
					if (activFunc == ActivationFunc::ReLU)
					{
						sigma = std::sqrt(2.0f / static_cast<float>(inRows));
					}
					else
					{
						sigma = std::sqrt(2.0f / static_cast<float>(inRows + outRows));
					}
					std::random_device rd;
					std::mt19937 mt(rd());
					std::normal_distribution dist(0.0f, sigma);

					Tensor weights(1, numOutputs, numInputs);
					for (size_t i = 0; i < weights.size(); i++)
					{
						weights[i] = dist(mt);
					}
					Tensor biases(1, numOutputs, 1);

					m_Layers.push_back(std::make_shared<FullyConnectedLayer>(std::move(weights), std::move(biases)));

					MakeInputOutputPair();
					MakeDeltaInputOutputPair();
				}

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				// Activation
				{
					m_Layers.push_back(std::make_shared<ActivationLayer>(activFunc));

					MakeInputOutputPair();
					MakeDeltaInputOutputPair();
				}
			}
			else if (std::holds_alternative<ConvolutionalDesc>(layerDesc))
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				ConvolutionalDesc convLayerDesc = std::get<ConvolutionalDesc>(layerDesc);
				size_t kernelChannels = convLayerDesc.NumKernels;
				size_t kernelRows = convLayerDesc.KernelWidth;
				size_t kernelCols = convLayerDesc.KernelHeight;
				ActivationFunc activFunc = convLayerDesc.ActivFunc;

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = kernelChannels;
				outRows = (inRows - kernelRows) + 1;
				outCols = (inCols - kernelCols) + 1;

				// Convolutional layer
				{
					float sigma = std::sqrt(2.0f / static_cast<float>(outChannels * kernelRows * kernelCols));
					std::random_device rd;
					std::mt19937 mt(rd());
					std::normal_distribution dist(0.0f, sigma);

					Tensor kernel(kernelChannels, kernelRows, kernelCols);
					for (size_t i = 0; i < kernel.size(); ++i)
					{
						kernel[i] = dist(mt);
					}

					m_Layers.push_back(std::make_shared<ConvolutionalLayer>(inChannels, outRows, outCols, kernel));

					MakeInputOutputPair();
					MakeDeltaInputOutputPair();
				}

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				// Activation
				{
					m_Layers.push_back(std::make_shared<ActivationLayer>(activFunc));

					MakeInputOutputPair();
					MakeDeltaInputOutputPair();
				}
			}
			else if (std::holds_alternative<PoolingDesc>(layerDesc))
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				PoolingDesc poolLayerDesc = std::get<PoolingDesc>(layerDesc);
				size_t tileWidth = poolLayerDesc.TileWidth;
				size_t tileHeight = poolLayerDesc.TileHeight;

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = outChannels;
				outRows = ((inRows - tileWidth) / tileWidth) + 1;
				outCols = ((inCols - tileHeight) / tileHeight) + 1;

				m_Layers.push_back(std::make_shared<MaxPoolingLayer>(tileWidth, tileHeight));

				MakeInputOutputPair();
				MakeDeltaInputOutputPair();
			}
			else if (std::holds_alternative<FlattenDesc>(layerDesc))
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				FlattenDesc flattenLayerDesc = std::get<FlattenDesc>(layerDesc);

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = 1;
				outRows = inChannels * inRows * inCols;
				outCols = 1;

				m_Layers.push_back(std::make_shared<FlattenLayer>());

				MakeInputOutputPair();
				MakeDeltaInputOutputPair();
			}
			else
			{
				MML_ASSERT(false, "Unhandled layer description!");
			}
		}
	}

	const Tensor &Sequential::feedForward(const Tensor &input)
	{
		dataInputAt(0) = input;

		for (auto it = m_Layers.begin(); it != m_Layers.end(); ++it)
		{
			std::shared_ptr<Layer> currentLayer = *it;
			size_t currIdx = static_cast<size_t>(it - m_Layers.begin());

			currentLayer->forward(
				dataInputAt(currIdx),
				dataOutputAt(currIdx));
		}

		return *m_Data.back().second;
	}

	float Sequential::feedBackward(const Tensor &expected)
	{
		size_t lastLayerIdx = m_Layers.size() - 1;
		size_t numOutputs = deltaOutputAt(lastLayerIdx).rows();
		float error = std::numeric_limits<float>::infinity();

		if (m_ObjectiveFunc == LossFunc::MSE)
		{
			Tensor::sub(dataOutputAt(lastLayerIdx), expected, deltaOutputAt(lastLayerIdx));
			error = Tensor::sumWith(deltaOutputAt(lastLayerIdx), [](float x) {
				return x * x;
			}) * (1.0f / static_cast<float>(numOutputs));
		}
		else if (m_ObjectiveFunc == LossFunc::CrossEntropy)
		{
			Tensor::zipWith(dataOutputAt(lastLayerIdx), expected, [](float x, float y) {
				return -y / x;
			}, deltaOutputAt(lastLayerIdx));
			error = -Tensor::sumWith(dataOutputAt(lastLayerIdx), expected, [](float x, float y) {
				return y * std::log(x);
			});			
		}

		for (auto it = m_Layers.rbegin(); it != m_Layers.rend(); ++it)
		{
			std::shared_ptr<Layer> currentLayer = *it;
			size_t currIdx = static_cast<size_t>(m_Layers.rend() - it - 1);

			currentLayer->backward(
				dataInputAt(currIdx),
				dataOutputAt(currIdx),
				deltaInputAt(currIdx),
				deltaOutputAt(currIdx));
			currentLayer->update(m_LearningRate);
		}

		return error;
	}

	const Tensor &Sequential::dataInputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].first);
	}

	const Tensor &Sequential::dataOutputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].second);
	}

	const Tensor &Sequential::deltaInputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].first);
	}

	const Tensor &Sequential::deltaOutputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].second);
	}

	Tensor &Sequential::dataInputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].first);
	}

	Tensor &Sequential::dataOutputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].second);
	}

	Tensor &Sequential::deltaInputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].first);
	}

	Tensor &Sequential::deltaOutputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].second);
	}
}