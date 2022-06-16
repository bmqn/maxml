#include "MmlLayer.h"
#include "MmlActivations.h"
#include "MmlLog.h"

#include "maxml/MmlTensor.h"

namespace maxml
{
	FullyConLayer::FullyConLayer(FTensor &&weights, FTensor &&biases)
		: DeltaWeights(weights.channels(), weights.rows(), weights.cols()), DeltaBiases(weights.channels(), weights.rows(), 1), Weights(std::forward<FTensor>(weights)), Biases(std::forward<FTensor>(biases))
	{
	}

	void FullyConLayer::forward(const FTensor &input, FTensor &output)
	{
		FTensor::matMult(Weights, input, output);
		FTensor::add(output, Biases, output);
	}

	void FullyConLayer::backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta)
	{
		FTensor::matMult(FTensor::transpose(Weights), outputDelta, inputDelta);
		FTensor::matMult(outputDelta, FTensor::transpose(input), DeltaWeights);
		FTensor::copy(outputDelta, DeltaBiases);
	}

	void FullyConLayer::update(float learningRate)
	{
		FTensor::aMinusXMultB(Weights, DeltaWeights, learningRate, Weights);
		FTensor::aMinusXMultB(Biases, DeltaBiases, learningRate, Biases);
	}

	ConvLayer::ConvLayer(size_t inChannels, size_t outRows, size_t outCols, const FTensor &kernel)
		: KernelChannels(kernel.channels()), KernelRows(kernel.rows()), KernelCols(kernel.cols()), KernelWindowed(inChannels, kernel.channels(), kernel.rows() * kernel.cols()), InputWindowed(inChannels, kernel.rows() * kernel.cols(), outRows * outCols), DeltaKernelWindowed(inChannels, kernel.channels(), kernel.rows() * kernel.cols()), DeltaInputWindowed(inChannels, kernel.rows() * kernel.cols(), outRows * outCols)
	{
		for (size_t iChan = 0; iChan < inChannels; ++iChan)
		{
			KernelWindowed.fill(iChan, FTensor::resize(kernel, 1, kernel.channels(), kernel.rows() * kernel.cols()));
		}
	}

	void ConvLayer::forward(const FTensor &input, FTensor &output)
	{
		for (size_t winRow = 0; winRow < InputWindowed.rows(); ++winRow)
		{
			for (size_t winCol = 0; winCol < InputWindowed.cols(); ++winCol)
			{
				size_t origRow = winCol % output.rows() + winRow / KernelRows;
				size_t origCol = winCol / output.cols() + winRow % KernelRows;

				for (size_t chan = 0; chan < input.channels(); ++chan)
				{
					InputWindowed(chan, winRow, winCol) = input(chan, origRow, origCol);
				}
			}
		}

		FTensor result = FTensor::matMult(KernelWindowed, InputWindowed);
		result.resize(output.channels(), output.rows(), output.cols());
		result.transpose();

		FTensor::copy(result, output);
	}

	void ConvLayer::backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta)
	{
		FTensor deltaOutputWindowed = FTensor::transpose(outputDelta);
		deltaOutputWindowed.resize(inputDelta.channels(), KernelChannels, outputDelta.rows() * outputDelta.cols());

		FTensor::matMult(deltaOutputWindowed, FTensor::transpose(InputWindowed), DeltaKernelWindowed);
		FTensor::matMult(FTensor::transpose(KernelWindowed), deltaOutputWindowed, DeltaInputWindowed);

		for (size_t winRow = 0; winRow < DeltaInputWindowed.rows(); ++winRow)
		{
			for (size_t winCol = 0; winCol < DeltaInputWindowed.cols(); ++winCol)
			{
				size_t origRow = winCol % output.rows() + winRow / KernelRows;
				size_t origCol = winCol / output.cols() + winRow % KernelRows;

				for (size_t chan = 0; chan < input.channels(); ++chan)
				{
					inputDelta(chan, origRow, origCol) = DeltaInputWindowed(chan, winRow, winCol);
				}
			}
		}
	}

	void ConvLayer::update(float learningRate)
	{
		FTensor::aMinusXMultB(KernelWindowed, DeltaKernelWindowed, learningRate, KernelWindowed);
	}

	MaxPoolLayer::MaxPoolLayer(size_t tileWidth, size_t tileHeight)
		: TileWidth(tileWidth), TileHeight(tileHeight)
	{
	}

	void MaxPoolLayer::forward(const FTensor &input, FTensor &output)
	{
		for (size_t iChan = 0; iChan < output.channels(); ++iChan)
		{
			for (size_t iRow = 0; iRow < output.rows(); ++iRow)
			{
				for (size_t iCol = 0; iCol < output.cols(); ++iCol)
				{
					float max = -std::numeric_limits<float>::infinity();

					for (size_t tRow = 0; tRow < TileWidth; ++tRow)
					{
						for (size_t tCol = 0; tCol < TileHeight; ++tCol)
						{
							float val = input(iChan, iRow * TileWidth + tRow, iCol * TileHeight + tCol);

							if (val > max)
							{
								max = val;
							}
						}
					}

					output(iChan, iRow, iCol) = max;
				}
			}
		}
	}

	void MaxPoolLayer::backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta)
	{
		inputDelta.fill(0.0);

		for (size_t iChan = 0; iChan < output.channels(); ++iChan)
		{
			for (size_t iRow = 0; iRow < output.rows(); ++iRow)
			{
				for (size_t iCol = 0; iCol < output.cols(); ++iCol)
				{
					float max = output(iChan, iRow, iCol);

					for (size_t tRow = 0; tRow < TileWidth; ++tRow)
					{
						for (size_t tCol = 0; tCol < TileHeight; ++tCol)
						{
							float val = input(iChan, iRow * TileWidth + tRow, iCol * TileHeight + tCol);

							if (val >= max)
							{
								inputDelta(iChan, iRow * TileWidth + tRow, iCol * TileHeight + tCol) = outputDelta(iChan, iRow, iCol);
								break;
							}
						}
					}
				}
			}
		}
	}

	void FlattenLayer::forward(const FTensor &input, FTensor &output)
	{
		FTensor::copy(input, output);
	}

	void FlattenLayer::backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta)
	{
		FTensor::copy(outputDelta, inputDelta);
	}

	ActvLayer::ActvLayer(ActivationFunc activation)
		: Activation(activation)
	{
	}

	void ActvLayer::forward(const FTensor &input, FTensor &output)
	{
		switch (Activation)
		{
		case ActivationFunc::Sigmoid:
			FTensor::fastSig(input, output);
			break;
		case ActivationFunc::Tanh:
			FTensor::mapWith(
				input, [](float x)
				{ return tanh(x); },
				output);
			break;
		case ActivationFunc::ReLU:
			FTensor::fastRelu(input, output);
			break;
		}
	}

	void ActvLayer::backward(const FTensor &input, const FTensor &output, FTensor &inputDelta, const FTensor &outputDelta)
	{
		switch (Activation)
		{
		case ActivationFunc::Sigmoid:
			FTensor::fastSigDeriv(output, inputDelta);
			FTensor::mult(inputDelta, outputDelta, inputDelta);
			break;
		case ActivationFunc::Tanh:
			FTensor::zipWith(
				input, outputDelta, [](float x, float y)
				{ return (tanhPrime(x)) * y; },
				inputDelta);
			break;
		case ActivationFunc::ReLU:
			FTensor::fastReluDeriv(input, inputDelta);
			FTensor::mult(inputDelta, outputDelta, inputDelta);
			break;
		}
	}
}