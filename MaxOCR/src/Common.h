#pragma once

inline float relu(float x)
{
	return x < 0.0f ? 0.0f : x;
}

inline float reluDerivative(float x)
{
	return x < 0.0f ? 0.0f : 1.0f;
}
