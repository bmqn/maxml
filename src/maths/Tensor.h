#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>

#include <assert.h>

#include <iostream>
#include <functional>

namespace mocr
{

	template <typename T>
	class Tensor
	{
	private:
		T *m_Data;

	public:
		int C; // Channels
		int W; // Width
		int H; // Height

		int Size;

		Tensor() : C(0), W(0), H(0), Size(0), m_Data(nullptr) {}

		Tensor(int c, int w, int h) : C(c), W(w), H(h), Size(c * w * h), m_Data(nullptr)
		{
			assert(Size > 0);

			m_Data = new T[Size];

			std::fill(m_Data, m_Data + Size, 0);

			// std::cout << "Creating tensor!" << std::endl;
		}

		Tensor(std::initializer_list<std::initializer_list<T>> data)
		{
			//std::cout << "Creating tensor!" << std::endl;

			C = 1;
			W = data.size();
			H = 0;

			if (W > 0)
				H = data.begin()->size();

			Size = W * H;

			assert(Size > 0);

			m_Data = new T[Size];

			int i = 0;
			for (auto &row : data)
			{
				std::copy(row.begin(), row.end(), m_Data + H * i);
				i++;
			}
		}

		~Tensor()
		{
			delete[] m_Data;

			//std::cout << "Deleting tensor!" << std::endl;
		}

		Tensor(const Tensor<T> &tensor) : C(tensor.C), W(tensor.W), H(tensor.H), Size(tensor.Size), m_Data(nullptr)
		{
			assert(Size > 0);

			m_Data = new T[Size];

			std::copy(tensor.m_Data, tensor.m_Data + Size, m_Data);

			// std::cout << "Copying tensor!" << std::endl;
		}

		Tensor(Tensor<T> &&tensor) : C(tensor.C), W(tensor.W), H(tensor.H), Size(tensor.Size), m_Data(tensor.m_Data)
		{
			tensor.C = 0;
			tensor.W = 0;
			tensor.H = 0;
			tensor.Size = 0;
			tensor.m_Data = nullptr;

			// std::cout << "Moving tensor!" << std::endl;
		}

		Tensor<T> &operator=(const Tensor<T> &tensor)
		{
			if (this == &tensor)
				return *this;

			if (Size != tensor.Size)
			{
				delete[] m_Data;
				m_Data = new T[tensor.Size];
			}

			C = tensor.C;
			W = tensor.W;
			H = tensor.H;
			Size = tensor.Size;

			std::copy(tensor.m_Data, tensor.m_Data + Size, m_Data);

			// std::cout << "Assigning (copying) tensor!" << std::endl;

			return *this;
		}

		Tensor<T> &operator=(Tensor<T> &&tensor)
		{
			if (this == &tensor)
				return *this;

			if (m_Data)
				delete[] m_Data;

			C = tensor.C;
			W = tensor.W;
			H = tensor.H;
			Size = tensor.Size;
			m_Data = tensor.m_Data;

			tensor.C = 0;
			tensor.W = 0;
			tensor.H = 0;
			tensor.Size = 0;
			tensor.m_Data = nullptr;

			//std::cout << "Assigning (moving) tensor!" << std::endl;

			return *this;
		}

		T &operator()(int c, int w, int h)
		{
			assert(c >= 0 && c < C);
			assert(w >= 0 && w < W);
			assert(h >= 0 && h < H);

			int index = c * (W * H) + w * (H) + h;
			return m_Data[index];
		}

		const T &operator()(int c, int w, int h) const
		{
			assert(c >= 0 && c < C);
			assert(w >= 0 && w < W);
			assert(h >= 0 && h < H);

			int index = c * (W * H) + w * (H) + h;
			return m_Data[index];
		}

		T &operator[](int i)
		{
			assert(i >= 0 && i < Size);

			return m_Data[i];
		}

		const T &operator[](int i) const
		{
			assert(i >= 0 && i < Size);

			return m_Data[i];
		}

		void fill(T val)
		{
			for (int i = 0; i < Size; i++)
				m_Data[i] = val;
		}

		void fill(int c, Tensor<T> &val)
		{
			assert(c >= 0 && c < C && val.W == W && val.H == val.H);

			for (int i = 0; i < W; i++)
				for (int j = 0; j < H; j++)
					this->operator()(c, i, j) = val(c, i, j);
		}

		std::string str() const
		{
			std::ostringstream ss;

			ss.precision(2);
			ss.fill(' ');

			int maxLen = 0;

			// EXPENSIVE: Finding max number width for alignment...
			for (int i = 0; i < Size; i++)
			{
				std::stringstream tss;
				tss << std::fixed << std::setprecision(2) << m_Data[i];

				std::string s;
				tss >> s;

				if (s.length() > maxLen)
					maxLen = s.length();
			}

			// Shape
			ss << "shp = (" << C << ", " << W << ", " << H << ")," << std::endl;

			// Array
			ss << "arr = ([";
			for (int c = 0; c < C; c++)
			{
				if (c > 0)
					ss << std::setw(9);
				ss << "[";
				for (int w = 0; w < W; w++)
				{
					if (w > 0)
						ss << std::setw(10);
					ss << "[";
					for (int h = 0; h < H; h++)
					{
						int index = c * (W * H) + w * (H) + h;

						// TODO: Temp for easy reading...
						if (m_Data[index] <= 0.0)
							ss << std::fixed << std::right << std::setw(maxLen) << "";
						else
							ss << std::fixed << std::right << std::setw(maxLen) << m_Data[index];
						if (h < H - 1)
							ss << ", ";
					}
					ss << "]";
					if (w < W - 1)
						ss << "," << std::endl;
				}
				ss << "]";
				if (c < C - 1)
					ss << "," << std::endl;
			}
			ss << "])";

			return ss.str();
		}
	};

	template <typename T>
	Tensor<T> add(const Tensor<T> &a, const Tensor<T> &b)
	{
		// TODO: size assertions

		Tensor<T> y(a.C, a.W, a.H);

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, i, j) + b(c, i, j);
				}

		return y;
	}

	template <typename T>
	void add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, i, j) + b(c, i, j);
				}
	}

	template <typename T>
	Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b)
	{
		// TODO: size assertions

		Tensor<T> y(a.C, a.W, a.H);

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, i, j) - b(c, i, j);
				}

		return y;
	}

	template <typename T>
	void sub(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, i, j) - b(c, i, j);
				}
	}

	template <typename T>
	Tensor<T> mult(const Tensor<T> &a, T s)
	{
		// TODO: size assertions

		Tensor<T> y(a.C, a.W, a.H);

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, i, j) * s;
				}

		return y;
	}

	template <typename T>
	Tensor<T> mult(const Tensor<T> &a, const Tensor<T> &b)
	{
		// TODO: size assertions

		Tensor<T> y(a.C, a.W, a.H);

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, i, j) * b(c, i, j);
				}

		return y;
	}

	template <typename T>
	Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b)
	{
		// assert(a.c_ == b.c_ && a.w_ == b.h_ && a.h_ == b.w_);
		assert(a.C == b.C && a.H == b.W);

		Tensor<T> y(a.C, a.W, b.H);

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					T sum{0};

					for (int k = 0; k < a.H; k++)
						sum += a(c, i, k) * b(c, k, j);

					y(c, i, j) = sum;
				}

		return y;
	}

	template <typename T>
	void matmul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					T sum{0};

					for (int k = 0; k < a.H; k++)
						sum += a(c, i, k) * b(c, k, j);

					y(c, i, j) = sum;
				}
	}

	template <typename T>
	Tensor<T> transpose(const Tensor<T> &a)
	{
		// TODO: size assertions

		Tensor<T> y(a.C, a.H, a.W);

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, j, i);
				}

		return y;
	}

	template <typename T>
	void transpose(const Tensor<T> &a, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.C; c++)
			for (int i = 0; i < y.W; i++)
				for (int j = 0; j < y.H; j++)
				{
					y(c, i, j) = a(c, j, i);
				}
	}

	template <typename T>
	T sum(const Tensor<T> &a)
	{
		// TODO: size assertions

		T sum{0};

		for (int c = 0; c < a.C; c++)
			for (int i = 0; i < a.W; i++)
				for (int j = 0; j < a.H; j++)
				{
					sum += a(c, i, j);
				}

		return sum;
	}

	template <typename T>
	Tensor<T> resize(const Tensor<T> &a, int c, int w, int h)
	{
		assert(c * w * h == a.Size);

		Tensor<T> y(a);

		y.C = c;
		y.W = w;
		y.H = h;

		return y;
	}

	template <typename T>
	Tensor<T> map(const Tensor<T> &a, std::function<T(T)> f)
	{
		Tensor<T> y(a.C, a.W, a.H);

		for (int i = 0; i < y.Size; i++)
		{
			y[i] = f(a[i]);
		}

		return y;
	}
}