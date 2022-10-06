#pragma once

namespace maxml
{
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

	template<typename VariantType, typename T, std::size_t index = 0>
	static constexpr std::size_t variantIndex()
	{
		static_assert(std::variant_size_v<VariantType> > index, "Type not found in variant");

		if constexpr (index == std::variant_size_v<VariantType>)
		{
			return index;
		}
		else if constexpr (std::is_same_v<std::variant_alternative_t<index, VariantType>, T>)
		{
			return index;
		}
		else
		{
			return variantIndex<VariantType, T, index + 1>();
		}
	}
}