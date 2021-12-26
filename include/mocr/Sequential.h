#ifndef H_SEQUENTIAL_H
#define H_SEQUENTIAL_H

#include "mocr/Tensor.h"

#include <vector>
#include <memory>

namespace mocr
{
	struct Layer;

	enum class ActivationFunc
	{
		SIGMOID,
		TANH,
		RELU
	};

	enum class LossFunc
	{
		MSE
	};

	class SequentialDesc
	{
	private:
		unsigned int m_InputChannels;
		unsigned int m_InputRows;
		unsigned int m_InputCols;
	};

	class Sequential
	{
	public:
		Sequential() = delete;
		Sequential(
			unsigned int inChannels,
			unsigned int inRows,
			unsigned int inCols,
			LossFunc objectiveFunc,
			double learningRate = 0.1
		);
		
		Sequential(const Sequential& other) = delete;
		Sequential(const Sequential&& other) = delete;
		Sequential& operator=(const Sequential& other) = delete;
		Sequential& operator=(const Sequential&& other) = delete;

		// TODO: Introduce a prototype object which describes and validates a network for
		//       construction
		void addFullyConnectedLayer(int connections, ActivationFunc activation);

		const DTensor& feedForward(const DTensor& input);
		double feedBackward(const DTensor& expected);

	private:
		const DTensor& dataInputAt(unsigned int index) const;
		const DTensor& dataOutputAt(unsigned int index) const;
		const DTensor& deltaInputAt(unsigned int index) const;
		const DTensor& deltaOutputAt(unsigned int index) const;

		DTensor& dataInputAt(unsigned int index);
		DTensor& dataOutputAt(unsigned int index);
		DTensor& deltaInputAt(unsigned int index);
		DTensor& deltaOutputAt(unsigned int index);

	private:
		unsigned int m_InChannels, m_InRows, m_InCols;

		// Each layer has a pair of tensors for input and output, respectively
		std::vector<std::pair<std::shared_ptr<DTensor>, std::shared_ptr<DTensor>>> m_Data;
		std::vector<std::pair<std::shared_ptr<DTensor>, std::shared_ptr<DTensor>>> m_Delta;
		std::vector< std::shared_ptr<Layer>> m_Layers;

		LossFunc m_ObjectiveFunc;

		// TODO: This should probably be associated with some kind of 'optimizer' object
		double m_LearningRate;
	};
}

#endif