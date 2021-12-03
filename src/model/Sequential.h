#ifndef H_SEQUENTIAL_H
#define H_SEQUENTIAL_H

#include "maths/Tensor.h"
#include "model/Layer.h"

#include <vector>
#include <memory>

namespace mocr
{
    class Sequential
    {
    public:
        Sequential() = delete;
        Sequential(unsigned int inChannels,
                   unsigned int inRows,
                   unsigned int inCols,
                   LossFunc objectiveFunc,
                   double learningRate = 0.1
        ) : m_InChannels(inChannels),
            m_InRows(inRows),
            m_InCols(inCols),
            m_ObjectiveFunc(objectiveFunc),
            m_LearningRate(learningRate)
        {            
        }

        void addFullyConnectedLayer(int connections, ActivationFunc activation);

        DTensor feedForward(DTensor &input);
        double feedBackward(const DTensor &expected);

    private:
        unsigned int m_InChannels, m_InRows, m_InCols;
        std::vector<std::shared_ptr<Layer>> m_Layers;
        LossFunc m_ObjectiveFunc;
        double m_LearningRate;
    };
}

#endif