#include "maxml/MmlSequential.h"

#include "MmlLayer.h"
#include "MmlLog.h"

#include <random>
#include <limits>

namespace maxml
{
	Sequential::Sequential(const SequentialDesc& sequentialDesc)
	{
		m_SequentialDesc = sequentialDesc;
		
		m_ObjectiveFunc = sequentialDesc.ObjectiveFunc;
		m_LearningRate = sequentialDesc.LearningRate;
		
		unsigned int prevChannels = 0, prevRows = 0, prevCols = 0;

		for (auto it = sequentialDesc.LayerDescs.begin(); it != sequentialDesc.LayerDescs.end(); it++)
		{
			SequentialDesc::VariantType layerDesc = *it;

			switch (SequentialDesc::getLayerKind(layerDesc.index()))
			{
				case LayerKind::Input:
				{
					MML_ASSERT(it == sequentialDesc.LayerDescs.begin());

					InputLayerDesc inpLayerDesc = std::get<InputLayerDesc>(layerDesc);

					prevChannels = inpLayerDesc.Channels;
					prevRows = inpLayerDesc.Rows;
					prevCols = inpLayerDesc.Cols;
					
					break;
				}
				case LayerKind::FullyConnected:
				{
					MML_ASSERT(it != sequentialDesc.LayerDescs.begin());

					FullyConnectedLayerDesc fcLayerDesc = std::get<FullyConnectedLayerDesc>(layerDesc);

					unsigned int numInputs   = prevRows;
					unsigned int numOutputs  = fcLayerDesc.NumOutputs;
					ActivationFunc activFunc = fcLayerDesc.ActivFunc;

					{
						std::shared_ptr<DTensor> input = m_Data.empty() ? nullptr : m_Data.back().second;
						std::shared_ptr<DTensor> output = std::make_shared<DTensor>(1, numOutputs, 1);
						m_Data.push_back(std::make_pair(input, output));

						std::shared_ptr<DTensor> inputDelta = m_Delta.empty() ? nullptr : m_Delta.back().second;
						std::shared_ptr<DTensor> outputDelta = std::make_shared<DTensor>(1, numOutputs, 1);
						m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

						std::random_device rd;
						std::mt19937 mt(rd());
						std::normal_distribution dist(0.0, 1.0);

						double scale = std::sqrt(1.0 / numInputs);

						DTensor weights(1, numOutputs, numInputs);
						DTensor biases(1, numOutputs, 1);

						for (int i = 0; i < weights.size(); i++)
						{
							weights[i] = dist(mt) * scale;
						}

						m_Layers.push_back(std::make_shared<FullyConLayer>(std::move(weights), std::move(biases)));
					}

					if (activFunc != ActivationFunc::None)
					{
						std::shared_ptr<DTensor> input = m_Data.back().second;
						std::shared_ptr<DTensor> output = std::make_shared<DTensor>(1, numOutputs, 1);
						m_Data.push_back(std::make_pair(input, output));

						std::shared_ptr<DTensor> inputDelta = m_Delta.back().second;
						std::shared_ptr<DTensor> outputDelta = std::make_shared<DTensor>(1, numOutputs, 1);
						m_Delta.push_back(std::make_pair(inputDelta, outputDelta));

						m_Layers.push_back(std::make_shared<ActvLayer>(1, numOutputs, 1, activFunc));
					}

					prevChannels = 1;
					prevRows = numOutputs;
					prevCols = 1;

					break;
				}
			}

			
		}
	}

	const DTensor& Sequential::feedForward(const DTensor &input)
	{
		InputLayerDesc inputLayerDesc = std::get<InputLayerDesc>(m_SequentialDesc.LayerDescs.front());

		MML_ASSERT(input.channels() == inputLayerDesc.Channels
			&& input.rows() == inputLayerDesc.Rows
			&& input.cols() == inputLayerDesc.Cols
		);

		for (auto it = m_Layers.begin(); it != m_Layers.end(); ++it)
		{
			auto currentLayer = *it;

			if (it == m_Layers.begin())
			{
				// TODO: The input for the first layer is always unused currently.
				//       It would be best to use some kind of special 'input' layer.
				currentLayer->forward(input, dataOutputAt(0));
			}
			else
			{
				size_t currIndex = static_cast<size_t>(it - m_Layers.begin());
				currentLayer->forward(dataInputAt(currIndex), dataOutputAt(currIndex));
			}
		}

		return *m_Data.back().second;
	}

	double Sequential::feedBackward(const DTensor &expected)
	{
		if (m_ObjectiveFunc == LossFunc::MSE)
		{
			// Mean Square Error ...
			DTensor::sub(dataOutputAt(m_Layers.size() - 1), expected, deltaOutputAt(m_Layers.size() - 1));

			for (auto it = m_Layers.rbegin(); it != m_Layers.rend() - 1; ++it)
			{
				auto currentLayer = *it;

				size_t currIndex = static_cast<size_t>(m_Layers.rend() - it - 1);
				currentLayer->backward(dataInputAt(currIndex), dataOutputAt(currIndex), deltaInputAt(currIndex), deltaOutputAt(currIndex));
				
				currentLayer->update(m_LearningRate);
			}

			size_t numOututs = deltaOutputAt(m_Layers.size() - 1).rows();
			auto error = DTensor::sumWith(
				deltaOutputAt(m_Layers.size() - 1),
				[](double x) { return x * x; }
			) * (1.0 / (double)numOututs);
			return error;
		}
		else
		{
			return std::numeric_limits<double>::infinity();
		}
	}

	const DTensor& Sequential::dataInputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].first);
	}

	const DTensor& Sequential::dataOutputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].second);
	}

	const DTensor& Sequential::deltaInputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].first);
	}

	const DTensor& Sequential::deltaOutputAt(size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].second);
	}

	DTensor& Sequential::dataInputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].first);
	}

	DTensor& Sequential::dataOutputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Data[index].second);
	}

	DTensor& Sequential::deltaInputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].first);
	}

	DTensor& Sequential::deltaOutputAt(size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Layers.size());

		return *(m_Delta[index].second);
	}
}