#ifndef H_SEQUENTIAL_H
#define H_SEQUENTIAL_H

#include "maxml/Tensor.h"

#include <vector>
#include <memory>
#include <variant>

namespace maxml
{
	/*
	* TODO: Describe a network with a json or xml file.
	*       See the json file below for an idea of how this might work.
	* 
	* 
	{
		"Name": "MnistNetwork",
		"LossFunc": "MeanSquared",
		"LearningRate": 0.3,
		"Layers": [
			{
				"Kind": "Input",
				"Description": {
					"Channels": 1,
					"Rows": 784,
					"Cols": 1
				}
			},
			{
				"Kind": "FullyConnected"
				"Description": {
					"NumOutputs": 256,
					"ActivationFunc": "Sigmoid"
				}
			},
			{
				"Kind": "FullyConnected"
				"Description": {
					"NumOutputs": 64,
					"ActivationFunc": "Sigmoid"
				}
			},
			{
				"Kind": "FullyConnected"
				"Description": {
					"NumOutputs": 10,
					"ActivationFunc": "Sigmoid"
				}
			}
		]
	}
	*/

	struct Layer;

	enum class ActivationFunc
	{
		None = -1,

		Sigmoid,
		Tanh,
		ReLU
	};

	enum class LossFunc
	{
		MSE
	};

	enum class LayerKind
	{
		Input,
		FullyConnected
	};

	struct InputLayerDesc
	{
		unsigned int Channels = 0;
		unsigned int Rows = 0;
		unsigned int Cols = 0;
	};

	struct FullyConnectedLayerDesc
	{
		unsigned int   NumOutputs = 0;
		ActivationFunc ActivFunc = ActivationFunc::None;
	};

	struct SequentialDesc
	{
		using VariantType = std::variant<
			InputLayerDesc,         // Input
			FullyConnectedLayerDesc // FullyConnected
		>;

		LossFunc                 ObjectiveFunc = LossFunc::MSE;
		double                   LearningRate = 0.1;
		std::vector<VariantType> LayerDescs = {};

		static LayerKind getLayerKind(std::size_t index)
		{
			// TODO: Some kind of way to verify the order here..
			//       maybe use macros in some way.
			return static_cast<LayerKind>(index);
		}
	};

	class Sequential
	{
	public:
		Sequential() = delete;
		Sequential(const SequentialDesc& sequentialDesc);

		Sequential(const Sequential& other) = delete;
		Sequential(const Sequential&& other) = delete;
		Sequential& operator=(const Sequential& other) = delete;
		Sequential& operator=(const Sequential&& other) = delete;

		const DTensor& feedForward(const DTensor& input);
		double feedBackward(const DTensor& expected);

	private:
		const DTensor& dataInputAt(std::size_t index) const;
		const DTensor& dataOutputAt(std::size_t index) const;
		const DTensor& deltaInputAt(std::size_t index) const;
		const DTensor& deltaOutputAt(std::size_t index) const;

		DTensor& dataInputAt(std::size_t index);
		DTensor& dataOutputAt(std::size_t index);
		DTensor& deltaInputAt(std::size_t index);
		DTensor& deltaOutputAt(std::size_t index);

	private:
		// Each layer has a pair of tensors for input and output, respectively
		std::vector<std::pair<std::shared_ptr<DTensor>, std::shared_ptr<DTensor>>> m_Data;
		std::vector<std::pair<std::shared_ptr<DTensor>, std::shared_ptr<DTensor>>> m_Delta;
		std::vector<std::shared_ptr<Layer>> m_Layers;

		// TODO: This should probably be specified with an 'output' layer
		LossFunc m_ObjectiveFunc;

		// TODO: This should probably be associated with some kind of 'optimizer' object
		double m_LearningRate;

		SequentialDesc m_SequentialDesc;
	};
}

#endif