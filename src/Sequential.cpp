#include "mocr/Sequential.h"

#include "Layer.h"

#include <random>
#include <limits>

namespace mocr
{
    Sequential::Sequential(
        unsigned int inChannels,
        unsigned int inRows,
        unsigned int inCols,
        LossFunc objectiveFunc,
        double learningRate
    ) : m_InChannels(inChannels),
        m_InRows(inRows),
        m_InCols(inCols),
        m_ObjectiveFunc(objectiveFunc),
        m_LearningRate(learningRate)
    {
    }

    const DTensor& Sequential::feedForward(const DTensor &input)
    {
        for (auto it = m_Layers.cbegin(); it != m_Layers.cend(); ++it)
        {
            auto currentLayer = *it;

            if (it == m_Layers.cbegin())
            {
                // TODO: The input for the first layer is always unused currently.
                //       It would be best to use some kind of special 'input' layer.
                currentLayer->forward(input, *m_Data[0].second);
            }
            else
            {
                unsigned int currIndex = it - m_Layers.cbegin();
                currentLayer->forward(*m_Data[currIndex].first, *m_Data[currIndex].second);
            }
        }

        return *m_Data.back().second;
    }

    double Sequential::feedBackward(const DTensor &expected)
    {
        if (m_ObjectiveFunc == LossFunc::MSE)
        {
            // Mean Square Error ...

            DTensor::sub(*m_Data.back().second, expected, *m_Deltas.back().second);

            for (auto it = m_Layers.crbegin(); it != m_Layers.crend() - 1; ++it)
            {
                auto currentLayer = *it;

                unsigned int currIndex = m_Layers.crend() - it - 1;
                currentLayer->backward(*m_Data[currIndex].first, *m_Data[currIndex].second, *m_Deltas[currIndex].first, *m_Deltas[currIndex].second);
                
                currentLayer->update(m_LearningRate);
            }

            auto error = DTensor::sumWith(*m_Deltas.back().second, [](double x) { return x * x; }) * 0.5;
            return error;
        }
        else
        {
            return std::numeric_limits<double>::infinity();
        }
    }

    void Sequential::addFullyConnectedLayer(int connections, ActivationFunc activation)
    {
        // TODO: validate the previous layer before allowing input

        // TODO: Introduce some kind of 'input' layer
        auto inputs  = m_Layers.size() == 0 ? m_InRows : m_Layers.back()->Rows;
        auto outputs = connections;

        if (m_Layers.size() == 0)
        {
            std::shared_ptr<DTensor> output = std::make_shared<DTensor>(1, outputs, 1);
            m_Data.push_back(std::make_pair(nullptr, output));

            std::shared_ptr<DTensor> outputDelta = std::make_shared<DTensor>(1, outputs, 1);
            m_Deltas.push_back(std::make_pair(nullptr , outputDelta));
        }
        else
        {
            std::shared_ptr<DTensor> input = m_Data.back().second;
            std::shared_ptr<DTensor> output = std::make_shared<DTensor>(1, outputs, 1);
            m_Data.push_back(std::make_pair(input, output));

            std::shared_ptr<DTensor> inputDelta = m_Deltas.back().second;
            std::shared_ptr<DTensor> outputDelta = std::make_shared<DTensor>(1, outputs, 1);
            m_Deltas.push_back(std::make_pair(inputDelta, outputDelta));
        }

        std::random_device rd;
        std::mt19937 mt(rd());
        std::normal_distribution dist(0.0, 1.0);
        double scale = std::sqrt(1.0 / inputs);
        DTensor weights(1, outputs, inputs);
        DTensor biases(1, outputs, 1);
        for (int i = 0; i < weights.size(); i++)
            weights[i] = dist(mt) * scale;

        m_Layers.push_back(std::make_shared<FullyConLayer>(std::move(weights), std::move(biases)));

        {
            std::shared_ptr<DTensor> input = m_Data.back().second;
            std::shared_ptr<DTensor> output = std::make_shared<DTensor>(1, outputs, 1);
            m_Data.push_back(std::make_pair(input, output));

            std::shared_ptr<DTensor> inputDelta = m_Deltas.back().second;
            std::shared_ptr<DTensor> outputDelta = std::make_shared<DTensor>(1, outputs, 1);
            m_Deltas.push_back(std::make_pair(inputDelta, outputDelta));
        }

        m_Layers.push_back(std::make_shared<ActvLayer>(1, connections, 1, activation));
    }
}