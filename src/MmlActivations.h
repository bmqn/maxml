#pragma once

#include <cmath>

namespace maxml
{
	// TODO: Optimise these functions.

	static float sig(float x)
	{
		return 1.0f / (1.0f + std::exp(-x));
	}

	static float sigPrime(float x)
	{
		return sig(x) * (1.0f - sig(x));
	}

	static float relu(float x)
	{
		return x < 0.0f ? 0.0f : x;
	}

	static float reluPrime(float x)
	{
		return x < 0.0f ? 0.0f : 1.0f;
	}

	static float tanh(float x)
	{
		return std::tanh(x);
	}

	static float tanhPrime(float x)
	{
		float coshx = std::cosh(x);

		return 1.0f / (coshx * coshx);
	}
}