#pragma once

#include <string>
#include <ostream>
#include <functional>

namespace maxml
{
	class Tensor
	{
	private:
		size_t m_Channels;
		size_t m_Rows;
		size_t m_Cols;
		size_t m_Size;

		float *m_Data;

	private:
		Tensor(size_t channels, size_t rows, size_t cols, float *data);

	public:
		Tensor();
		Tensor(size_t channels, size_t rows, size_t cols);
		Tensor(std::initializer_list<float> data);
		Tensor(std::initializer_list<std::initializer_list<float>> data);
		Tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> data);
		Tensor(const Tensor &tensor);
		Tensor(Tensor &&tensor) noexcept;

		~Tensor();

		Tensor &operator=(const Tensor &tensor);
		Tensor &operator=(Tensor &&tensor) noexcept;

		float &operator()(size_t channel, size_t row, size_t col);
		const float &operator()(size_t channel, size_t row, size_t col) const;

		float &operator[](size_t index);
		const float &operator[](size_t index) const;

		size_t size() const;
		size_t channels() const;
		size_t rows() const;
		size_t cols() const;

		void fill(float val);
		void fill(size_t channel, const Tensor &val);
		void resize(size_t channels, size_t rows, size_t cols);
		void transpose();

		std::string str() const;

	public:
		static Tensor resize(const Tensor &a, size_t channels, size_t rows, size_t cols);

		static Tensor add(const Tensor &a, const Tensor &b);
		static void add(const Tensor &a, const Tensor &b, Tensor &y);

		static Tensor sub(const Tensor &a, const Tensor &b);
		static void sub(const Tensor &a, const Tensor &b, Tensor &y);

		static Tensor mult(const Tensor &a, float s);
		static Tensor mult(const Tensor &a, float s, Tensor &y);
		static Tensor mult(const Tensor &a, const Tensor &b);
		static void mult(const Tensor &a, const Tensor &b, Tensor &y);

		static Tensor matMult(const Tensor &a, const Tensor &b);
		static void matMult(const Tensor &a, const Tensor &b, Tensor &y);

		static Tensor transpose(const Tensor &a);
		static void transpose(const Tensor &a, Tensor &y);

		static float sum(const Tensor &a);
		static float sumWith(const Tensor &a, std::function<float(float)> f);

		static Tensor mapWith(const Tensor &a, std::function<float(float)> f);
		static void mapWith(const Tensor &a, std::function<float(float)> f, Tensor &y);

		static void zipWith(const Tensor &a, const Tensor &b, std::function<float(float, float)> f, Tensor &y);

		static void aMinusXMultB(const Tensor &a, const Tensor &b, float x, Tensor &y);
		static void fastSig(const Tensor &a, Tensor &y);
		static void fastRelu(const Tensor &a, Tensor &y);

		static void copy(const Tensor &a, Tensor &y);
	};

	// std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
	// {
	// 	os << tensor.str();
	// 	return os;
	// }
}