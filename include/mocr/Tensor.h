#pragma once

#include "mocr/Assert.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <functional>

namespace mocr
{
	template<typename T>
	class Tensor;

	using DTensor = Tensor<double>;
	using FTensor = Tensor<float>;
	using ITensor = Tensor<int>;

	template <typename T>
	class Tensor
	{
	private:
		int m_Channels;
		int m_Rows;
		int m_Cols;
		int m_Size;

		T* m_Data;

		// TODO: Is this a good way to handle ownership ?
		//       Perhaps we should have a TensorView instead...
		bool m_Owned;
	
	private:
		Tensor(int chnls, int rows, int cols, T* data)
			: m_Channels(chnls)
			, m_Rows(rows)
			, m_Cols(cols)
			, m_Size(chnls * rows * cols)
			, m_Data(data)
			, m_Owned(false)
		{
		}

	public:
		Tensor()
			: m_Channels(0)
			, m_Rows(0)
			, m_Cols(0)
			, m_Size(0)
			, m_Data(nullptr)
			, m_Owned(false)
		{}

		Tensor(int chnls, int rows, int cols)
			: m_Channels(chnls)
			, m_Rows(rows)
			, m_Cols(cols)
			, m_Size(chnls * rows * cols)
			, m_Data(nullptr)
		{
			MOCR_ASSERT(m_Size > 0);

			m_Data  = new T[m_Size];
			m_Owned = true;
			
			std::fill(m_Data, m_Data + m_Size, T{ 0 });
		}

		Tensor(std::initializer_list<T> data) : m_Channels(1), m_Rows(static_cast<int>(data.size())), m_Cols(1), m_Size(static_cast<int>(data.size()))
		{
			MOCR_ASSERT(m_Size > 0);

			m_Data = new T[m_Size];
			m_Owned = true;
			
			std::copy(data.begin(), data.end(), m_Data);
		}

		Tensor(std::initializer_list<std::initializer_list<T>> data)
		{
			m_Channels = 1;
			m_Rows = data.size();
			m_Cols = 0;
			
			if (m_Rows > 0)
			{
				m_Cols = data.begin()->size();
			}

			m_Size = m_Rows * m_Cols;
			
			MOCR_ASSERT(m_Size > 0);
			
			m_Data = new T[m_Size];
			m_Owned = true;

			auto i = 0;
			for (auto& row : data)
			{
				std::copy(row.begin(), row.end(), m_Data + m_Cols * i);
				i++;
			}
		}

		Tensor(const Tensor<T>& tensor)
			: m_Channels(tensor.m_Channels)
			, m_Rows(tensor.m_Rows)
			, m_Cols(tensor.m_Cols)
			, m_Size(tensor.m_Size)
			, m_Data(nullptr)
		{
			if (tensor.m_Owned)
			{
				m_Data = new T[m_Size];
				m_Owned = true;

				std::copy(tensor.m_Data, tensor.m_Data + m_Size, m_Data);
			}
			else
			{
				m_Data = tensor.m_Data;
				m_Owned = false;
			}
		}

		Tensor(Tensor<T>&& tensor) noexcept
			: m_Channels(tensor.m_Channels)
			, m_Rows(tensor.m_Rows)
			, m_Cols(tensor.m_Cols)
			, m_Size(tensor.m_Size)
			, m_Data(tensor.m_Data)
			, m_Owned(tensor.m_Owned)
		{
			// TODO: If m_Owned is false then we can move without having to
			//       'zero' out the other tensor ?

			tensor.m_Channels = 0;
			tensor.m_Rows = 0;
			tensor.m_Cols = 0;
			tensor.m_Size = 0;
			tensor.m_Data = nullptr;
			tensor.m_Owned = false;
		}

		~Tensor()
		{
			if (m_Owned)
			{
				delete[] m_Data;
			}
		}

		Tensor<T>& operator=(const Tensor<T>& tensor)
		{
			if (this == &tensor)
			{
				return *this;
			}

			if (m_Size != tensor.m_Size)
			{
				if (m_Owned)
				{
					delete[] m_Data;
				}

				m_Data = new T[tensor.m_Size];
				m_Owned = true;
			}

			m_Channels = tensor.m_Channels;
			m_Rows = tensor.m_Rows;
			m_Cols = tensor.m_Cols;
			m_Size = tensor.m_Size;

			std::copy(tensor.m_Data, tensor.m_Data + m_Size, m_Data);

			return *this;
		}

		Tensor<T>& operator=(Tensor<T>&& tensor)
		{
			if (this == &tensor)
			{
				return *this;
			}

			if (m_Data && m_Owned)
			{
				delete[] m_Data;
			}

			m_Channels = tensor.m_Channels;
			m_Rows = tensor.m_Rows;
			m_Cols = tensor.m_Cols;
			m_Size = tensor.m_Size;
			m_Data = tensor.m_Data;
			m_Owned = tensor.m_Owned;

			tensor.m_Channels = 0;
			tensor.m_Rows = 0;
			tensor.m_Cols = 0;
			tensor.m_Size = 0;
			tensor.m_Data = nullptr;
			tensor.m_Owned = false;

			return *this;
		}

		Tensor<T> operator()(int chnl) const
		{
			MOCR_ASSERT(chnl >= 0 && chnl < m_Channels);

			auto index = chnl * (m_Rows * m_Cols);
			return Tensor<T>(1, m_Rows, m_Cols, &m_Data[index]);
		}

		T& operator()(int chnl, int row, int col)
		{
			MOCR_ASSERT(chnl >= 0 && chnl < m_Channels
				&& row >= 0 && row < m_Rows
				&& col >= 0 && col < m_Cols);

			auto index = chnl * (m_Rows * m_Cols) + row * (m_Cols) + col;
			return m_Data[index];
		}

		const T& operator()(int chnl, int row, int col) const
		{
			MOCR_ASSERT(chnl >= 0 && chnl < m_Channels
				&& row >= 0 && row < m_Rows
				&& col >= 0 && col < m_Cols);

			auto index = chnl * (m_Rows * m_Cols) + row * (m_Cols)+col;
			return m_Data[index];
		}

		T& operator[](int index)
		{
			MOCR_ASSERT(index >= 0 && index < m_Size);

			return m_Data[index];
		}

		const T& operator[](int index) const
		{
			MOCR_ASSERT(index >= 0 && index < m_Size);

			return m_Data[index];
		}

		int size() const
		{
			return m_Size;
		}

		int channels() const
		{
			return m_Channels;
		}

		int rows() const
		{
			return m_Rows;
		}

		int cols() const
		{
			return m_Cols;
		}

		void fill(T val)
		{
			for (auto i = 0; i < m_Size; i++)
			{
				m_Data[i] = val;
			}
		}

		void fill(int chnl, Tensor<T>& val)
		{
			MOCR_ASSERT(chnl >= 0 && chnl < m_Channels
				&& val.m_Channels == m_Channels
				&& val.m_Rows == m_Rows
				&& val.m_Cols == m_Cols
			);

			for (auto i = 0; i < m_Rows; i++)
			{
				for (auto j = 0; j < m_Cols; j++)
				{
					this->operator()(chnl, i, j) = val(chnl, i, j);
				}
			}
		}

		void resize(int c, int w, int h)
		{
			int newSize = c * w * h;

			MOCR_ASSERT(newSize > 0);

			if (newSize == m_Size)
			{
				m_Channels = c;
				m_Rows = w;
				m_Cols = h;
			}
			else
			{
				m_Channels = c;
				m_Rows = w;
				m_Cols = h;
				m_Size = newSize;

				if (m_Owned)
				{
					delete[] m_Data;
				}

				m_Data = new T[m_Size];
				m_Owned = true;

				std::fill(m_Data, m_Data + m_Size, T{ 0 });
			}
		}

		std::string str() const
		{
			std::ostringstream ss;

			ss.precision(2);
			ss.fill(' ');

			auto maxLen = 0;

			// EXPENSIVE: Finding max number width for alignment...
			for (auto i = 0; i < m_Size; i++)
			{
				std::stringstream tss;
				tss << std::fixed << std::setprecision(2) << m_Data[i];

				std::string s;
				tss >> s;

				if (s.length() > maxLen)
					maxLen = static_cast<int>(s.length());
			}

			// Shape
			ss << "shp = (" << m_Channels << ", " << m_Rows << ", " << m_Cols << ")," << std::endl;

			// Array
			ss << "arr = ([";
			for (auto c = 0; c < m_Channels; c++)
			{
				if (c > 0)
					ss << std::setw(9);
				ss << "[";
				for (auto w = 0; w < m_Rows; w++)
				{
					if (w > 0)
						ss << std::setw(10);
					ss << "[";
					for (auto h = 0; h < m_Cols; h++)
					{
						int index = c * (m_Rows * m_Cols) + w * (m_Cols)+h;

						// TODO: Temp for easy reading...
						if (false) // m_Data[index] <= 0.0)
							ss << std::fixed << std::right << std::setw(maxLen) << "";
						else
							ss << std::fixed << std::right << std::setw(maxLen) << m_Data[index];
						if (h < m_Cols - 1)
							ss << ", ";
					}
					ss << "]";
					if (w < m_Rows - 1)
						ss << "," << std::endl;
				}
				ss << "]";
				if (c < m_Channels - 1)
					ss << "," << std::endl;
			}
			ss << "])";

			return ss.str();
		}

		static Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
			);

			Tensor<T> y(a.m_Channels, a.m_Rows, a.m_Cols);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) + b(c, i, j);
					}

			return y;
		}

		static void add(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
				&& y.m_Channels == a.m_Channels
				&& y.m_Rows == a.m_Rows
				&& y.m_Cols == a.m_Cols
			);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) + b(c, i, j);
					}
		}

		static Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
			);

			Tensor<T> y(a.m_Channels, a.m_Rows, a.m_Cols);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) - b(c, i, j);
					}

			return y;
		}

		static void sub(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
				&& y.m_Channels == a.m_Channels
				&& y.m_Rows == a.m_Rows
				&& y.m_Cols == a.m_Cols
			);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) - b(c, i, j);
					}
		}

		static Tensor<T> mult(const Tensor<T>& a, T s)
		{
			Tensor<T> y(a.m_Channels, a.m_Rows, a.m_Cols);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) * s;
					}

			return y;
		}

		static Tensor<T> mult(const Tensor<T>& a, T s, Tensor<T>& y)
		{
			MOCR_ASSERT(y.m_Channels == a.m_Channels
				&& y.m_Rows == a.m_Rows
				&& y.m_Cols == a.m_Cols
			);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) * s;
					}

			return y;
		}

		static Tensor<T> mult(const Tensor<T>& a, const Tensor<T>& b)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
			);

			Tensor<T> y(a.m_Channels, a.m_Rows, a.m_Cols);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) * b(c, i, j);
					}

			return y;
		}

		static void mult(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
				&& y.m_Channels == a.m_Channels
				&& y.m_Rows == a.m_Rows
				&& y.m_Cols == a.m_Cols
			);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, i, j) * b(c, i, j);
					}
		}

		static Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Cols == b.m_Rows
			);

			Tensor<T> y(a.m_Channels, a.m_Rows, b.m_Cols);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						T sum{ 0 };

						for (auto k = 0; k < a.m_Cols; k++)
							sum += a(c, i, k) * b(c, k, j);

						y(c, i, j) = sum;
					}

			return y;
		}

		static void matmul(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y, bool transposeA = false, bool transposeB = false)
		{
			// TODO: Handle case when both are true
			MOCR_ASSERT(!(transposeA && transposeB));

			if (transposeA)
			{
				MOCR_ASSERT(a.m_Channels == b.m_Channels
					&& a.m_Rows == b.m_Rows
					&& y.m_Channels == a.m_Channels
					&& y.m_Rows == a.m_Cols
					&& y.m_Cols == b.m_Cols
				);

				for (auto c = 0; c < y.m_Channels; c++)
					for (auto i = 0; i < y.m_Rows; i++)
						for (auto j = 0; j < y.m_Cols; j++)
						{
							T sum{ 0 };

							for (auto k = 0; k < a.m_Rows; k++)
							{
								sum += a(c, k, i) * b(c, k, j);
							}

							y(c, i, j) = sum;
						}
			}
			else if (transposeB)
			{
				MOCR_ASSERT(a.m_Channels == b.m_Channels
					&& a.m_Cols == b.m_Cols
					&& y.m_Channels == a.m_Channels
					&& y.m_Rows == a.m_Rows
					&& y.m_Cols == b.m_Rows
				);

				for (auto c = 0; c < y.m_Channels; c++)
					for (auto i = 0; i < y.m_Rows; i++)
						for (auto j = 0; j < y.m_Cols; j++)
						{
							T sum{ 0 };

							for (auto k = 0; k < a.m_Cols; k++)
							{
								sum += a(c, i, k) * b(c, j, k);
							}

							y(c, i, j) = sum;
						}
			}
			else
			{
				MOCR_ASSERT(a.m_Channels == b.m_Channels
					&& a.m_Cols == b.m_Rows
					&& y.m_Channels == a.m_Channels
					&& y.m_Rows == a.m_Rows
					&& y.m_Cols == b.m_Cols
				);

				for (auto c = 0; c < y.m_Channels; c++)
					for (auto i = 0; i < y.m_Rows; i++)
						for (auto j = 0; j < y.m_Cols; j++)
						{
							T sum{ 0 };

							for (auto k = 0; k < a.m_Cols; k++)
							{
								sum += a(c, i, k) * b(c, k, j);
							}

							y(c, i, j) = sum;
						}
			}
		}

		static Tensor<T> transpose(const Tensor<T>& a)
		{
			Tensor<T> y(a.m_Channels, a.m_Cols, a.m_Rows);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, j, i);
					}

			return y;
		}

		static void transpose(const Tensor<T>& a, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Channels == y.m_Channels
				&& a.m_Rows == y.m_Cols
				&& a.m_Cols == y.m_Rows
			);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = a(c, j, i);
					}
		}

		static T sum(const Tensor<T>& a)
		{
			T sum{ 0 };

			for (auto i = 0; i < a.m_Size; i++)
			{
				sum += a[i];
			}

			return sum;
		}

		static T sumWith(const Tensor<T>& a, std::function<T(T)> f)
		{
			T sum{ 0 };

			for (auto i = 0; i < a.m_Size; i++)
			{
				sum += f(a[i]);
			}

			return sum;
		}

		static Tensor<T> map(const Tensor<T>& a, std::function<T(T)> f)
		{
			Tensor<T> y(a.m_Channels, a.m_Rows, a.m_Cols);

			for (auto i = 0; i < y.m_Size; i++)
			{
				y[i] = f(a[i]);
			}

			return y;
		}

		static void map(const Tensor<T>& a, std::function<T(T)> f, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Size == y.m_Size);

			for (auto i = 0; i < y.m_Size; i++)
			{
				y[i] = f(a[i]);
			}
		}

		static void zip(const Tensor<T>& a, const Tensor<T>& b, std::function<T(T, T)> f, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Channels == b.m_Channels
				&& a.m_Rows == b.m_Rows
				&& a.m_Cols == b.m_Cols
				&& y.m_Channels == a.m_Channels
				&& y.m_Rows == a.m_Rows
				&& y.m_Cols == a.m_Cols
			);

			for (auto c = 0; c < y.m_Channels; c++)
				for (auto i = 0; i < y.m_Rows; i++)
					for (auto j = 0; j < y.m_Cols; j++)
					{
						y(c, i, j) = f(a(c, i, j), b(c, i, j));
					}
		}

		static void copy(const Tensor<T>& a, Tensor<T>& y)
		{
			MOCR_ASSERT(a.m_Size == y.m_Size);

			for (auto i = 0; i < y.m_Size; i++)
			{
				y[i] = a[i];
			}
		}
	};
}