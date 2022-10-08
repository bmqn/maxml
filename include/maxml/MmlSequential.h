#pragma once

#include "maxml/MmlTensor.h"

#include <vector>
#include <variant>
#include <memory>

namespace maxml
{
	/*
	* TODO: Describe a network with a json or xml file.
	* See the json file below for an idea of how this might work.
	*

	{
		"Name": "MnistNetwork",
		"LossFunc": "MSE",
		"LearningRate": 0.3,
		"Layers": [
			{
				"Kind": "Input",
				"Description": {
					"Channels": 1,
					"Rows": 28,
					"Cols": 28
				}
			},
			{
				"Kind": "Flatten"
				"Description": {
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

	enum class ActivationFunc : uint32_t
	{
		None = 0,
		Sigmoid = 1,
		Tanh = 2,
		ReLU = 3,
		Softmax = 4
	};

	enum class PoolingFunc : uint32_t
	{
		Max = 0
	};

	enum class LossFunc : uint32_t
	{
		MSE = 0,
		CrossEntropy = 1
	};

	enum class LayerKind : uint32_t
	{
		Input = 0,
		FullyConnected = 1,
		Convolutional = 2,
		Polling = 3,
		Flatten = 4
	};

	struct InputDesc
	{
		uint64_t Channels = 0;
		uint64_t Rows = 0;
		uint64_t Cols = 0;
	};

	struct FullyConnectedDesc
	{
		uint64_t NumOutputs = 0;
		ActivationFunc ActivFunc = ActivationFunc::None;
	};

	struct ConvolutionalDesc
	{
		uint64_t NumKernels = 8;
		uint64_t KernelWidth = 3;
		uint64_t KernelHeight = 3;
		ActivationFunc ActivFunc = ActivationFunc::None;
	};

	struct PoolingDesc
	{
		uint64_t TileWidth = 2;
		uint64_t TileHeight = 2;
		PoolingFunc PoolFunc = PoolingFunc::Max;
	};

	struct FlattenDesc
	{
	};

	struct SequentialDesc
	{
		using LayerDesc = std::variant<
			InputDesc,          // Input
			FullyConnectedDesc, // FullyConnected
			ConvolutionalDesc,  // Convolutional
			PoolingDesc,        // Pooling
			FlattenDesc         // Flatten
		>;

		LossFunc ObjectiveFunc = LossFunc::MSE;
		float LearningRate = 0.1f;
		std::vector<LayerDesc> LayerDescs = {};
	};

	InputDesc makeInput(size_t channels, size_t rows, size_t cols);
	FullyConnectedDesc makeFullyConnected(size_t numOutputs, ActivationFunc activFunc);
	ConvolutionalDesc makeConvolutional(size_t numKernels, size_t kernelWidth, size_t kernelHeight, ActivationFunc activFunc);
	PoolingDesc makePooling(size_t tileWidth, size_t tileHeight, PoolingFunc poolFunc);
	FlattenDesc makeFlatten();

	class Sequential
	{
	public:
		Sequential() = delete;
		Sequential(const SequentialDesc &description);
		Sequential(const std::string &path);

		Sequential(const Sequential &other) = delete;
		Sequential(const Sequential &&other) = delete;
		Sequential &operator=(const Sequential &other) = delete;
		Sequential &operator=(const Sequential &&other) = delete;

		const Tensor &feedForward(const Tensor &input);
		float feedBackward(const Tensor &expected);

		void save(const std::string &path);

	private:
		void construct(const std::string &path);
		void construct(const SequentialDesc &description);

		const Tensor &dataInputAt(size_t index) const;
		const Tensor &dataOutputAt(size_t index) const;
		const Tensor &deltaInputAt(size_t index) const;
		const Tensor &deltaOutputAt(size_t index) const;

		Tensor &dataInputAt(size_t index);
		Tensor &dataOutputAt(size_t index);
		Tensor &deltaInputAt(size_t index);
		Tensor &deltaOutputAt(size_t index);

	private:
		// Each layer has a pair of tensors for input and output, respectively.
		// The output and input of two sequential layers point to the same tensor.
		std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> m_Data;
		std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> m_Delta;
		std::vector<std::shared_ptr<Layer>> m_Layers;

		SequentialDesc m_Description;
	};
}