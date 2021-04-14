#ifndef H_ACTIVATIONS_H
#define H_ACTIVATIONS_H

#include <cmath>

namespace mocr
{
    static double sig(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static double sigPrime(double x)
    {
        return sig(x) * (1.0 - sig(x));
    }

    static double relu(double x)
    {
        return x < 0.0 ? 0.0 : x;
    }

    static double reluPrime(double x)
    {
        return x < 0.0 ? 0.0 : 1.0;
    }

    static double tanh(double x)
    {
        return std::tanh(x);
    }

    static double tanhPrime(double x)
    {
        return 1.0 / (std::cosh(x) * std::cosh(x));
    }
}

#endif