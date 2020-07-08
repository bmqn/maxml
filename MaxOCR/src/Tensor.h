#pragma once

#include <iostream>
#include <iomanip>

template<typename T>
struct Tensor
{
	std::shared_ptr<T[]> data;
	int sX, sY, sZ;

	Tensor() = delete;

	Tensor(int x, int y, int z) : sX(x), sY(y), sZ(z)
	{
		data = std::make_unique<T[]>(x * y * z);
	}

	Tensor(const Tensor<T>& other) : sX(other.sX), sY(other.sY), sZ(other.sZ)
	{
		data = std::make_unique<T[]>(x * y * z);
		memcpy(this->data.get(), other.data.get(), other.sX * other.sY * other.sZ * sizeof(T));
	}

	/*const Tensor<T>& operator= (const Tensor<T>& other)
	{
		if (this == &other)
			return *this;

		if (sX * sY * sZ != other.sX * other.sY * other.sZ)
		{
			delete[] data;
			data = new T[other.sX * other.sY * other.sZ];
		}

		sX = other.sX;
		sY = other.sY;
		sZ = other.sZ;

		memcpy(this->data, other.data, other.sX * other.sY * other.sZ * sizeof(T));
		
		return *this;
	}*/

	T& operator()(int x, int y, int z)
	{
		return get(x, y, z);
	}

	const T& operator()(int x, int y, int z) const
	{
		return get(x, y, z);
	}

	T& get(int x, int y, int z)
	{
		return data[index(x, y, z)];
	}

	const T& get(int x, int y, int z) const
	{
		return data[index(x, y, z)];
	}

	int index(int x, int y, int z) const
	{
		return x * (sZ * sY) + y * (sZ)+ z;
	}

	friend std::ostream& operator<< (std::ostream& out, const Tensor& obj)
	{
		out << "[";
		for (int x = 0; x < obj.sX; x++)
		{
			if (x > 0) out << std::setfill(' ') << std::setw(2);
			out << "[";
			for (int y = 0; y < obj.sY; y++)
			{
				if (y > 0) out << std::setfill(' ') << std::setw(3);
				out << "[";
				for (int z = 0; z < obj.sZ; z++)
				{
					out << std::fixed << std::setprecision(2) << obj.get(x, y, z);
					if (z < obj.sZ - 1) out << ", " << std::setw(5);
				}
				out << "]";
				if (y < obj.sY - 1) out << "," << std::endl;
			}
			out << "]";
			if (x < obj.sX - 1) out << "," << std::endl << std::endl;
		}
		out << "]";

		return out;
	}
};

