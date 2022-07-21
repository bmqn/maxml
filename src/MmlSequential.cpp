#include "maxml/MmlSequential.h"

#include "MmlLayer.h"
#include "MmlLog.h"

#include <random>
#include <limits>

namespace maxml
{
	InputLayerDesc makeInput(size_t channels, size_t rows, size_t cols)
	{
		return { channels, rows, cols };
	}

	FullConLayerDesc makeFullCon(size_t numOutputs, ActivationFunc activFunc)
	{
		return { numOutputs, activFunc };
	}

	ConvLayerDesc makeConv(size_t numKernels, size_t kernelWidth, size_t kernelHeight, ActivationFunc activFunc)
	{
		return { numKernels, kernelWidth, kernelHeight, activFunc };
	}

	PoolLayerDesc makePool(size_t tileWidth, size_t tileHeight, PoolingFunc poolFunc)
	{
		return { tileWidth, tileHeight, poolFunc };
	}

	FlattenLayerDesc makeFlatten()
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

		for (auto it = sequentialDesc.LayerDescs.begin(); it != sequentialDesc.LayerDescs.end(); it++)
		{
			SequentialDesc::LayerDesc layerDesc = *it;

			switch (SequentialDesc::getLayerKind(layerDesc.index()))
			{
			case LayerKind::Input:
			{
				MML_ASSERT(it == sequentialDesc.LayerDescs.begin(), "Cannot have more than one input layer!");

				InputLayerDesc inpLayerDesc = std::get<InputLayerDesc>(layerDesc);

				inChannels = outChannels = inpLayerDesc.Channels;
				inRows = outRows = inpLayerDesc.Rows;
				inCols = outCols = inpLayerDesc.Cols;

				break;
			}
			case LayerKind::FullyConnected:
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				FullConLayerDesc fcLayerDesc = std::get<FullConLayerDesc>(layerDesc);

				size_t numInputs = outRows;
				size_t numOutputs = fcLayerDesc.NumOutputs;
				ActivationFunc activFunc = fcLayerDesc.ActivFunc;

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = 1;
				outRows = numOutputs;
				outCols = 1;

				{
					std::shared_ptr<Tensor> input = m_Data.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Data.back().second;
					std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Data.push_back(std::make_pair(input, output));

					std::shared_ptr<Tensor> inputDelta = m_Delta.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Delta.back().second;
					std::shared_ptr<Tensor> outputDelta = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

					std::random_device rd;
					std::mt19937 mt(rd());
					float sigma;
					if (activFunc == ActivationFunc::ReLU)
					{
						sigma = std::sqrt(2.0f / static_cast<float>(inRows));
					}
					else
					{
						sigma = std::sqrt(2.0f / static_cast<float>(inRows + outRows));
					}
					std::normal_distribution dist(0.0f, sigma);

					Tensor weights(1, numOutputs, numInputs);
					Tensor biases(1, numOutputs, 1);

					for (size_t i = 0; i < weights.size(); i++)
					{
						weights[i] = dist(mt);
					}

					m_Layers.push_back(std::make_shared<FullyConLayer>(std::move(weights), std::move(biases)));
				}

				{
					std::shared_ptr<Tensor> input = m_Data.back().second;
					std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Data.push_back(std::make_pair(input, output));

					std::shared_ptr<Tensor> inputDelta = m_Delta.back().second;
					std::shared_ptr<Tensor> outputDelta = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

					m_Layers.push_back(std::make_shared<ActvLayer>(activFunc));
				}

				break;
			}
			case LayerKind::Convolutional:
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				ConvLayerDesc convLayerDesc = std::get<ConvLayerDesc>(layerDesc);

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

				{
					std::shared_ptr<Tensor> input = m_Data.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Data.back().second;
					std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Data.push_back(std::make_pair(input, output));

					std::shared_ptr<Tensor> inputDelta = m_Delta.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Delta.back().second;
					std::shared_ptr<Tensor> outputDelta = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

					std::random_device rd;
					std::mt19937 mt(rd());
					float sigma = std::sqrt(2.0f / static_cast<float>(outChannels * kernelRows * kernelCols));
					std::normal_distribution dist(0.0f, sigma);

					Tensor kernel(kernelChannels, kernelRows, kernelCols);

					for (size_t i = 0; i < kernel.size(); ++i)
					{
						kernel[i] = dist(mt);
					}

					m_Layers.push_back(std::make_shared<ConvLayer>(inChannels, outRows, outCols, kernel));
				}

				{
					std::shared_ptr<Tensor> input = m_Data.back().second;
					std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Data.push_back(std::make_pair(input, output));

					std::shared_ptr<Tensor> inputDelta = m_Delta.back().second;
					std::shared_ptr<Tensor> outputDelta = std::make_shared<Tensor>(outChannels, outRows, outCols);
					m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

					m_Layers.push_back(std::make_shared<ActvLayer>(activFunc));
				}

				break;
			}
			case LayerKind::Polling:
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				PoolLayerDesc poolLayerDesc = std::get<PoolLayerDesc>(layerDesc);
				MML_ASSERT(poolLayerDesc.PoolFunc != PoolingFunc::Average, "Average pooling is not implemented yet!");

				size_t tileWidth = poolLayerDesc.TileWidth;
				size_t tileHeight = poolLayerDesc.TileHeight;

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = outChannels;
				outRows = ((inRows - tileWidth) / tileWidth) + 1;
				outCols = ((inCols - tileHeight) / tileHeight) + 1;

				std::shared_ptr<Tensor> input = m_Data.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Data.back().second;
				std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outChannels, outRows, outCols);
				m_Data.push_back(std::make_pair(input, output));

				std::shared_ptr<Tensor> inputDelta = m_Delta.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Delta.back().second;
				std::shared_ptr<Tensor> outputDelta = std::make_shared<Tensor>(outChannels, outRows, outCols);
				m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

				m_Layers.push_back(std::make_shared<MaxPoolLayer>(tileWidth, tileHeight));

				break;
			}
			case LayerKind::Flatten:
			{
				MML_ASSERT(it != sequentialDesc.LayerDescs.begin(), "Must start with an input layer!");

				FlattenLayerDesc flattenLayerDesc = std::get<FlattenLayerDesc>(layerDesc);

				inChannels = outChannels;
				inRows = outRows;
				inCols = outCols;

				outChannels = 1;
				outRows = inChannels * inRows * inCols;
				outCols = 1;

				std::shared_ptr<Tensor> input = m_Data.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Data.back().second;
				std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outChannels, outRows, outCols);
				m_Data.push_back(std::make_pair(input, output));

				std::shared_ptr<Tensor> inputDelta = m_Delta.empty() ? std::make_shared<Tensor>(inChannels, inRows, inCols) : m_Delta.back().second;
				std::shared_ptr<Tensor> outputDelta = std::make_shared<Tensor>(outChannels, outRows, outCols);
				m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

				m_Layers.push_back(std::make_shared<FlattenLayer>());

				break;
			}
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