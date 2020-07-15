#pragma once

#include <vector>
#include <array>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream> 

template<int... Args>
constexpr int packProd()
{
	return ( Args * ... );
}

template<typename T, int ... Dims>
struct Tensor
{
	int size;
	std::array<int, sizeof...(Dims)> dimensions;
	std::vector<T> data;

	Tensor()
		: size(packProd<Dims ...>()), dimensions({ Dims ... })
	{
		data.resize(size);
	}

	int getIndex(std::array<int, sizeof...(Dims)> index) const
	{
		constexpr int Dim = sizeof...(Dims);

		int pos = index[0];
		for (int dim = 1; dim < Dim; dim++)
			pos = pos * this->dimensions[dim] + index[dim];

		return pos;
	}

	template <typename ... Inds>
	T& operator() (Inds... indices)
	{
		constexpr int Dim = sizeof...(Inds);
		static_assert(Dim == sizeof...(Dims));

		return data[getIndex({ indices... })];
	}

	T& operator[] (int index)
	{
		return data[index];
	}

	const T operator[] (int index) const
	{
		return data[index];
	}

	friend std::ostream& operator<< (std::ostream& out, const Tensor& obj)
	{
		out << obj.toString();
		return out;
	}

	std::string toString() const
	{
		constexpr int Dim = sizeof...(Dims);

		std::ostringstream ss;
		std::array<int, Dim> index { 0 };
		bool exit = false;

		do 
		{
			for (int j = Dim - 1; j >= 0; j--)
			{
				if (index[j] > 0)
					break;
				else if (index[j] == 0)
					ss << "[";
			}

			ss << std::fixed << std::setprecision(2) << std::right << std::setw(5) << this->data[this->getIndex(index)];

			index[Dim - 1] += 1;

			for (int j = Dim - 1; j >= 0; j--)
			{
				if (index[j] < this->dimensions[j])
					break;
				else if (index[j] == this->dimensions[j])
				{
					// Then inc at j-1 and make all <= j = 0;
					if (j > 0) index[j - 1] += 1;
					else exit = true;

					for (int k = j; k < Dim; k++)
						index[k] = 0;
				}
			}

			for (int j = Dim - 1; j >= 0; j--)
			{
				if (index[j] > 0)
				{
					if (j == Dim - 1) ss << ", ";
					if (j < Dim - 1) ss << ", " << std::endl << std::string(j + 1, ' ');
					break;
				}
				else if (index[j] == 0)
					ss << "]";
			}

		} while (!exit);


		return ss.str();
	}
};

