#include "maxml/MmlTensor.h"

#include "MmlLog.h"
#include "MmlActivations.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <functional>

#include <immintrin.h>

namespace maxml
{
	Tensor::Tensor(size_t channels, size_t rows, size_t cols, float *data)
		: m_Channels(channels), m_Rows(rows), m_Cols(cols), m_Size(channels * rows * cols), m_Data(data)
	{
	}

	Tensor::Tensor()
		: m_Channels(0), m_Rows(0), m_Cols(0), m_Size(0), m_Data(nullptr)
	{
	}

	Tensor::Tensor(size_t channels, size_t rows, size_t cols)
		: m_Channels(channels), m_Rows(rows), m_Cols(cols), m_Size(channels * rows * cols), m_Data(nullptr)
	{
		m_Data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
		MML_ASSERT(m_Data != nullptr, "Failed to allocate memory for tensor!");

		std::memset(m_Data, 0, m_Size * sizeof(float));
	}

	Tensor::Tensor(std::initializer_list<float> data)
		: m_Channels(1), m_Rows(data.size()), m_Cols(1), m_Size(m_Channels * m_Rows * m_Cols), m_Data(nullptr)
	{
		m_Data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
		MML_ASSERT(m_Data != nullptr, "Failed to allocate memory for tensor!");

		std::copy(data.begin(), data.end(), m_Data);
	}

	Tensor::Tensor(std::initializer_list<std::initializer_list<float>> data)
		: m_Channels(1), m_Rows(data.size()), m_Cols(data.begin()->size()), m_Size(m_Channels * m_Rows * m_Cols), m_Data(nullptr)
	{
		m_Data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
		MML_ASSERT(m_Data != nullptr, "Failed to allocate memory for tensor!");

		size_t i = 0;
		for (const auto &row : data)
		{
			std::copy(row.begin(), row.end(), m_Data + i * m_Cols);
			i++;
		}
	}
	
	Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> data)
		: m_Channels(data.size()), m_Rows(data.begin()->size()), m_Cols(data.begin()->begin()->size()), m_Size(m_Channels * m_Rows * m_Cols), m_Data(nullptr)
	{
		m_Data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
		MML_ASSERT(m_Data != nullptr, "Failed to allocate memory for tensor!");

		size_t i = 0, j = 0;
		for (const auto &channel : data)
		{
			for (const auto &row : channel)
			{
				std::copy(row.begin(), row.end(), m_Data + i * (m_Rows * m_Cols) + j * m_Cols);
				j++;
			}
			j = 0;
			i++;
		}
	}

	Tensor::Tensor(const Tensor &tensor)
		: m_Channels(tensor.m_Channels), m_Rows(tensor.m_Rows), m_Cols(tensor.m_Cols), m_Size(tensor.m_Size), m_Data(nullptr)
	{
		m_Data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
		MML_ASSERT(m_Data != nullptr, "Failed to allocate memory for tensor!");

		std::copy(tensor.m_Data, tensor.m_Data + m_Size, m_Data);
	}

	Tensor::Tensor(Tensor &&tensor) noexcept
		: m_Channels(tensor.m_Channels), m_Rows(tensor.m_Rows), m_Cols(tensor.m_Cols), m_Size(tensor.m_Size), m_Data(tensor.m_Data)
	{
		tensor.m_Channels = 0;
		tensor.m_Rows = 0;
		tensor.m_Cols = 0;
		tensor.m_Size = 0;
		tensor.m_Data = nullptr;
	}

	Tensor::~Tensor()
	{
		_mm_free(m_Data);
	}

	Tensor &Tensor::operator=(const Tensor &tensor)
	{
		if (this == &tensor)
		{
			return *this;
		}

		m_Channels = tensor.m_Channels;
		m_Rows = tensor.m_Rows;
		m_Cols = tensor.m_Cols;

		if (m_Size != tensor.m_Size)
		{
			_mm_free(m_Data);

			m_Size = tensor.m_Size;

			m_Data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
			MML_ASSERT(m_Data != nullptr, "Failed to allocate memory for tensor!");
		}

		std::copy(tensor.m_Data, tensor.m_Data + tensor.m_Size, m_Data);

		return *this;
	}

	Tensor &Tensor::operator=(Tensor &&tensor) noexcept
	{
		MML_ASSERT(this != &tensor);

		_mm_free(m_Data);

		m_Channels = tensor.m_Channels;
		m_Rows = tensor.m_Rows;
		m_Cols = tensor.m_Cols;
		m_Size = tensor.m_Size;
		m_Data = tensor.m_Data;

		tensor.m_Channels = 0;
		tensor.m_Rows = 0;
		tensor.m_Cols = 0;
		tensor.m_Size = 0;
		tensor.m_Data = nullptr;

		return *this;
	}

	float &Tensor::operator()(size_t channel, size_t row, size_t col)
	{
		MML_ASSERT(channel >= 0 && channel < m_Channels && row >= 0 && row < m_Rows && col >= 0 && col < m_Cols);

		size_t index = channel * (m_Rows * m_Cols) + row * m_Cols + col;
		return m_Data[index];
	}

	const float &Tensor::operator()(size_t channel, size_t row, size_t col) const
	{
		MML_ASSERT(channel >= 0 && channel < m_Channels && row >= 0 && row < m_Rows && col >= 0 && col < m_Cols);

		size_t index = channel * (m_Rows * m_Cols) + row * m_Cols + col;
		return m_Data[index];
	}

	float &Tensor::operator[](size_t index)
	{
		MML_ASSERT(index >= 0 && index < m_Size);

		return m_Data[index];
	}

	const float &Tensor::operator[](size_t index) const
	{
		MML_ASSERT(index >= 0 && index < m_Size);

		return m_Data[index];
	}

	size_t Tensor::size() const
	{
		return m_Size;
	}

	size_t Tensor::channels() const
	{
		return m_Channels;
	}

	size_t Tensor::rows() const
	{
		return m_Rows;
	}

	size_t Tensor::cols() const
	{
		return m_Cols;
	}

	void Tensor::fill(float val)
	{
		std::fill(m_Data, m_Data + m_Size, val);
	}

	void Tensor::fill(size_t channel, const Tensor &val)
	{
		MML_ASSERT(channel >= 0 && channel < m_Channels && val.m_Channels == 1 && val.m_Rows == m_Rows && val.m_Cols == m_Cols);

		std::copy(val.m_Data, val.m_Data + val.m_Size, &m_Data[channel * (m_Rows * m_Cols)]);
	}

	void Tensor::resize(size_t channels, size_t rows, size_t cols)
	{
		size_t size = channels * rows * cols;
		MML_ASSERT(size > 0, "Tensor cannot be zero-sized!");

		m_Channels = channels;
		m_Rows = rows;
		m_Cols = cols;

		if (size != m_Size)
		{
			float *data = reinterpret_cast<float *>(_mm_malloc(size * sizeof(float), 32));
			MML_ASSERT(data != nullptr, "Failed to allocate memory for tensor!");

			std::copy(m_Data, m_Data + std::min(m_Size, size), data);

			_mm_free(m_Data);

			m_Size = size;
			m_Data = data;
		}
	}

	void Tensor::transpose()
	{
		float *data = reinterpret_cast<float *>(_mm_malloc(m_Size * sizeof(float), 32));
		MML_ASSERT(data != nullptr, "Failed to allocate memory for tensor!");

		for (size_t c = 0; c < m_Channels; c++)
		{
			const float *a_c = &m_Data[c * (m_Rows * m_Cols)];
			float *y_c = &data[c * (m_Rows * m_Cols)];

			for (size_t k = 0; k < m_Rows * m_Cols; ++k)
			{
				size_t i = k / m_Cols;
				size_t j = k % m_Cols;
				y_c[k] = a_c[m_Rows * j + i];
			}
		}

		_mm_free(m_Data);

		std::swap(m_Rows, m_Cols);
		m_Data = data;
	}

	std::string Tensor::str() const
	{
		std::ostringstream ss;

		ss.precision(2);
		ss.fill(' ');

		size_t maxLen = 0;

		// EXPENSIVE: Finding max number width for alignment...
		for (size_t i = 0; i < m_Size; i++)
		{
			std::stringstream tss;
			tss << std::fixed << std::setprecision(2) << m_Data[i];

			std::string s;
			tss >> s;

			if (s.length() > maxLen)
			{
				maxLen = s.length();
			}
		}

		// Shape
		ss << "shp = (" << m_Channels << ", " << m_Rows << ", " << m_Cols << ")," << std::endl;

		// Array
		ss << "arr = ([";
		for (size_t c = 0; c < m_Channels; c++)
		{
			if (c > 0)
			{
				ss << std::setw(9);
			}
			ss << "[";
			for (size_t w = 0; w < m_Rows; w++)
			{
				if (w > 0)
				{
					ss << std::setw(10);
				}
				ss << "[";
				for (size_t h = 0; h < m_Cols; h++)
				{
					size_t index = c * (m_Rows * m_Cols) + w * m_Cols + h;

					// TODO: Temp for easy reading...
					if (false) // m_Data[index] <= 0.0)
					{
						ss << std::fixed << std::right << std::setw(maxLen) << "";
					}
					else
					{
						ss << std::fixed << std::right << std::setw(maxLen) << m_Data[index];
					}
					if (h < m_Cols - 1)
					{
						ss << ", ";
					}
				}
				ss << "]";
				if (w < m_Rows - 1)
				{
					ss << "," << std::endl;
				}
			}
			ss << "]";
			if (c < m_Channels - 1)
			{
				ss << "," << std::endl;
			}
		}
		ss << "])";

		return ss.str();
	}

	Tensor Tensor::resize(const Tensor &a, size_t channels, size_t rows, size_t cols)
	{
		size_t size = channels * rows * cols;
		MML_ASSERT(size > 0, "Tensor cannot be zero-sized!");

		float *data = reinterpret_cast<float *>(_mm_malloc(size * sizeof(float), 32));
		MML_ASSERT(data != nullptr, "Failed to allocate memory for tensor!");

		std::memset(data, 0, size * sizeof(float));
		std::copy(a.m_Data, a.m_Data + std::min(a.m_Size, size), data);

		return Tensor(channels, rows, cols, data);
	}

	Tensor Tensor::add(const Tensor &a, const Tensor &b)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols);

		Tensor y(a.m_Channels, a.m_Rows, a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 resultv = _mm256_add_ps(av, bv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] + b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] + b.m_Data[i];
			}
		}

		return y;
	}

	void Tensor::add(const Tensor &a, const Tensor &b, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols && y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 resultv = _mm256_add_ps(av, bv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] + b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] + b.m_Data[i];
			}
		}
	}

	Tensor Tensor::sub(const Tensor &a, const Tensor &b)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols);

		Tensor y(a.m_Channels, a.m_Rows, a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 resultv = _mm256_sub_ps(av, bv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] - b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] - b.m_Data[i];
			}
		}

		return y;
	}

	void Tensor::sub(const Tensor &a, const Tensor &b, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols && y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 resultv = _mm256_sub_ps(av, bv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] - b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] - b.m_Data[i];
			}
		}
	}

	Tensor Tensor::mult(const Tensor &a, float s)
	{
		Tensor y(a.m_Channels, a.m_Rows, a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			const __m256 sv = _mm256_set1_ps(s);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 resultv = _mm256_mul_ps(av, sv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] * s;
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] * s;
			}
		}

		return y;
	}

	Tensor Tensor::mult(const Tensor &a, float s, Tensor &y)
	{
		MML_ASSERT(y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			const __m256 sv = _mm256_set1_ps(s);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 resultv = _mm256_mul_ps(av, sv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] * s;
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] * s;
			}
		}

		return y;
	}

	Tensor Tensor::mult(const Tensor &a, const Tensor &b)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols);

		Tensor y(a.m_Channels, a.m_Rows, a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 resultv = _mm256_mul_ps(av, bv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] * b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] * b.m_Data[i];
			}
		}

		return y;
	}

	void Tensor::mult(const Tensor &a, const Tensor &b, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols && y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 resultv = _mm256_mul_ps(av, bv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] * b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] * b.m_Data[i];
			}
		}
	}

	Tensor Tensor::matMult(const Tensor &a, const Tensor &b)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Cols == b.m_Rows);

		Tensor y(a.m_Channels, a.m_Rows, b.m_Cols);

		float *b_ckj = reinterpret_cast<float *>(_mm_malloc(a.m_Cols * sizeof(float), 32));
		MML_ASSERT(b_ckj != nullptr, "Failed to allocate memory for tensor!");

		for (size_t c = 0; c < y.m_Channels; c++)
		{
			const float *a_c = &a.m_Data[c * (a.m_Rows * a.m_Cols)];
			const float *b_c = &b.m_Data[c * (b.m_Rows * b.m_Cols)];
			float *y_c = &y.m_Data[c * (y.m_Rows * y.m_Cols)];

			for (size_t j = 0; j < y.m_Cols; j++)
			{
				for (size_t k = 0; k < a.m_Cols; k++)
				{
					b_ckj[k] = b_c[k * b.m_Cols + j];
				}

				for (size_t i = 0; i < y.m_Rows; i++)
				{
					const float *a_cik = &a_c[i * a.m_Cols];

					float sum{};

					if (a.m_Cols >= 8)
					{
						MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
						MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);

						for (size_t k = 0; k < a.m_Cols - 7; k += 8)
						{
							__m256 av = _mm256_load_ps(a_cik + k);
							__m256 bv = _mm256_load_ps(b_ckj + k);
							__m256 abv = _mm256_mul_ps(av, bv);

							__m128 hiQuadv = _mm256_extractf128_ps(abv, 1);
							__m128 loQuadv = _mm256_castps256_ps128(abv);
							__m128 sumQuadv = _mm_add_ps(loQuadv, hiQuadv);
							__m128 loDualv = sumQuadv;
							__m128 hiDualv = _mm_movehl_ps(sumQuadv, sumQuadv);
							__m128 sumDualv = _mm_add_ps(loDualv, hiDualv);
							__m128 lov = sumDualv;
							__m128 hiv = _mm_shuffle_ps(sumDualv, sumDualv, 0x1);
							__m128 sumv = _mm_add_ss(lov, hiv);

							sum += _mm_cvtss_f32(sumv);
						}

						for (size_t k = a.m_Cols - 7; k < a.m_Cols; k++)
						{
							sum += a_cik[k] * b_ckj[k];
						}
					}
					else
					{
						for (size_t k = 0; k < a.m_Cols; k++)
						{
							sum += a_cik[k] * b_ckj[k];
						}
					}

					y_c[i * y.m_Cols + j] = sum;
				}
			}
		}

		_mm_free(b_ckj);

		return y;
	}

	void Tensor::matMult(const Tensor &a, const Tensor &b, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Cols == b.m_Rows);

		float *b_ckj = reinterpret_cast<float *>(_mm_malloc(a.m_Cols * sizeof(float), 32));
		MML_ASSERT(b_ckj != nullptr, "Failed to allocate memory for tensor!");

		for (size_t c = 0; c < y.m_Channels; c++)
		{
			const float *a_c = &a.m_Data[c * (a.m_Rows * a.m_Cols)];
			const float *b_c = &b.m_Data[c * (b.m_Rows * b.m_Cols)];
			float *y_c = &y.m_Data[c * (y.m_Rows * y.m_Cols)];

			for (size_t j = 0; j < y.m_Cols; j++)
			{
				for (size_t k = 0; k < a.m_Cols; k++)
				{
					b_ckj[k] = b_c[k * b.m_Cols + j];
				}

				for (size_t i = 0; i < y.m_Rows; i++)
				{
					const float *a_cik = &a_c[i * a.m_Cols];

					float sum{};

					if (a.m_Cols >= 8)
					{
						MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
						MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);

						for (size_t k = 0; k < a.m_Cols - 7; k += 8)
						{
							__m256 av = _mm256_load_ps(a_cik + k);
							__m256 bv = _mm256_load_ps(b_ckj + k);
							__m256 abv = _mm256_mul_ps(av, bv);

							__m128 hiQuadv = _mm256_extractf128_ps(abv, 1);
							__m128 loQuadv = _mm256_castps256_ps128(abv);
							__m128 sumQuadv = _mm_add_ps(loQuadv, hiQuadv);
							__m128 loDualv = sumQuadv;
							__m128 hiDualv = _mm_movehl_ps(sumQuadv, sumQuadv);
							__m128 sumDualv = _mm_add_ps(loDualv, hiDualv);
							__m128 lov = sumDualv;
							__m128 hiv = _mm_shuffle_ps(sumDualv, sumDualv, 0x1);
							__m128 sumv = _mm_add_ss(lov, hiv);

							sum += _mm_cvtss_f32(sumv);
						}

						for (size_t k = a.m_Cols - 7; k < a.m_Cols; k++)
						{
							sum += a_cik[k] * b_ckj[k];
						}
					}
					else
					{
						for (size_t k = 0; k < a.m_Cols; k++)
						{
							sum += a_cik[k] * b_ckj[k];
						}
					}

					y_c[i * y.m_Cols + j] = sum;
				}
			}
		}

		_mm_free(b_ckj);
	}

	
	Tensor Tensor::transpose(const Tensor &a)
	{
		Tensor y(a.m_Channels, a.m_Cols, a.m_Rows);

		for (size_t c = 0; c < y.m_Channels; c++)
		{
			const float *a_c = &a.m_Data[c * (a.m_Rows * a.m_Cols)];
			float *y_c = &y.m_Data[c * (y.m_Rows * y.m_Cols)];

			for (size_t k = 0; k < y.m_Rows * y.m_Cols; ++k)
			{
				size_t i = k / y.m_Cols;
				size_t j = k % y.m_Cols;
				y_c[k] = a_c[y.m_Rows * j + i];
			}
		}

		return y;
	}

	
	void Tensor::transpose(const Tensor &a, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == y.m_Channels && a.m_Rows == y.m_Cols && a.m_Cols == y.m_Rows);

		for (size_t c = 0; c < y.m_Channels; c++)
		{
			const float *a_c = &a.m_Data[c * (a.m_Rows * a.m_Cols)];
			float *y_c = &y.m_Data[c * (y.m_Rows * y.m_Cols)];

			for (size_t k = 0; k < y.m_Rows * y.m_Cols; ++k)
			{
				size_t i = k / y.m_Cols;
				size_t j = k % y.m_Cols;
				y_c[k] = a_c[y.m_Rows * j + i];
			}
		}
	}

	
	float Tensor::sum(const Tensor &a)
	{
		float sum{0};

		for (size_t i = 0; i < a.m_Size; i++)
		{
			sum += a.m_Data[i];
		}

		return sum;
	}

	
	float Tensor::sumWith(const Tensor &a, std::function<float(float)> f)
	{
		float sum{0};

		for (size_t i = 0; i < a.m_Size; i++)
		{
			sum += f(a.m_Data[i]);
		}

		return sum;
	}

	
	Tensor Tensor::mapWith(const Tensor &a, std::function<float(float)> f)
	{
		Tensor y(a.m_Channels, a.m_Rows, a.m_Cols);

		for (size_t i = 0; i < y.m_Size; i++)
		{
			y.m_Data[i] = f(a.m_Data[i]);
		}

		return y;
	}

	
	void Tensor::mapWith(const Tensor &a, std::function<float(float)> f, Tensor &y)
	{
		MML_ASSERT(a.m_Size == y.m_Size);

		for (size_t i = 0; i < y.m_Size; i++)
		{
			y.m_Data[i] = f(a.m_Data[i]);
		}
	}

	
	void Tensor::zipWith(const Tensor &a, const Tensor &b, std::function<float(float, float)> f, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols && y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		for (size_t i = 0; i < y.m_Size; ++i)
		{
			y.m_Data[i] = f(a.m_Data[i], b.m_Data[i]);
		}
	}

	void Tensor::aMinusXMultB(const Tensor &a, const Tensor &b, float x, Tensor &y)
	{
		MML_ASSERT(a.m_Channels == b.m_Channels && a.m_Rows == b.m_Rows && a.m_Cols == b.m_Cols && y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(b.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			__m256 xv = _mm256_set1_ps(x);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 bv = _mm256_load_ps(b.m_Data + i);
				__m256 xbv = _mm256_mul_ps(xv, bv);
				__m256 resultv = _mm256_sub_ps(av, xbv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] - x * b.m_Data[k];
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] - x * b.m_Data[i];
			}
		}
	}

	void Tensor::fastSig(const Tensor &a, Tensor &y)
	{
		MML_ASSERT(y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			static const __m256 onev = _mm256_set1_ps(1.0f);
			static const __m256 halfv = _mm256_set1_ps(0.5f);
			static const __m256 sign_mask = _mm256_set1_ps(-0.f);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 halfav = _mm256_mul_ps(av, halfv);
				__m256 absav = _mm256_andnot_ps(sign_mask, av);
				__m256 oneplusabsav = _mm256_add_ps(onev, absav);
				__m256 quotientv = _mm256_div_ps(halfav, oneplusabsav);
				__m256 resultv = _mm256_add_ps(quotientv, halfv);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = (0.5f * a.m_Data[k]) / (1 + std::abs(a.m_Data[k])) + 0.5f;
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = (0.5f * a.m_Data[i]) / (1 + std::abs(a.m_Data[i])) + 0.5f;
			}
		}
	}

	void Tensor::fastSigDeriv(const Tensor &a, Tensor &y)
	{
		MML_ASSERT(y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			static const __m256 onev = _mm256_set1_ps(1.0f);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 oneminusav = _mm256_sub_ps(onev, av);
				__m256 resultv = _mm256_mul_ps(av, oneminusav);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] * (1.0f - a.m_Data[k]);
			}
		}
		else
		{
			for (size_t i = 0; i < y.m_Size; ++i)
			{
				y.m_Data[i] = a.m_Data[i] * (1.0f - a.m_Data[i]);
			}
		}
	}

	void Tensor::fastRelu(const Tensor &a, Tensor &y)
	{
		MML_ASSERT(y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		if (y.m_Size >= 8)
		{
			MML_ASSERT(((uintptr_t)(a.m_Data) & 31) == 0);
			MML_ASSERT(((uintptr_t)(y.m_Data) & 31) == 0);

			static const __m256 zerov = _mm256_set1_ps(0.0f);

			for (size_t i = 0; i < y.m_Size - 7; i += 8)
			{
				__m256 av = _mm256_load_ps(a.m_Data + i);
				__m256 resultv = _mm256_max_ps(av, zerov);

				_mm256_store_ps(y.m_Data + i, resultv);
			}

			for (size_t k = y.m_Size - 7; k < y.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] < 0.0f ? 0.0f : a.m_Data[k];
			}
		}
		else
		{
			for (size_t k = 0; k < a.m_Size; k++)
			{
				y.m_Data[k] = a.m_Data[k] < 0.0f ? 0.0f : a.m_Data[k];
			}
		}
	}

	void Tensor::fastReluDeriv(const Tensor &a, Tensor &y)
	{
		MML_ASSERT(y.m_Channels == a.m_Channels && y.m_Rows == a.m_Rows && y.m_Cols == a.m_Cols);

		for (size_t k = 0; k < a.m_Size; k++)
		{
			y.m_Data[k] = a.m_Data[k] < 0.0f ? 0.0f : 1.0f;
		}
	}

	
	void Tensor::copy(const Tensor &a, Tensor &y)
	{
		MML_ASSERT(a.m_Size == y.m_Size);
		std::copy(a.m_Data, a.m_Data + a.m_Size, y.m_Data);
	}
}