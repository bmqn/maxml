#pragma once

#include "../src/MmlLog.h"

#include <string>
#include <ostream>
#include <functional>

namespace maxml
{
	template <typename T>
	class Tensor;

	using DTensor = Tensor<double>;
	using FTensor = Tensor<float>;
	using ITensor = Tensor<int>;

	template <typename T>
	class Tensor
	{
	private:
		size_t m_Channels;
		size_t m_Rows;
		size_t m_Cols;
		size_t m_Size;

		T *m_Data;

	private:
		Tensor(size_t channels, size_t rows, size_t cols, T *data);

	public:
		Tensor();
		Tensor(size_t channels, size_t rows, size_t cols);
		Tensor(std::initializer_list<T> data);
		Tensor(std::initializer_list<std::initializer_list<T>> data);
		Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> data);
		Tensor(const Tensor &tensor);
		Tensor(Tensor &&tensor) noexcept;

		~Tensor();

		Tensor &operator=(const Tensor &tensor);
		Tensor &operator=(Tensor &&tensor) noexcept;

		T &operator()(size_t channel, size_t row, size_t col);
		const T &operator()(size_t channel, size_t row, size_t col) const;

		T &operator[](size_t index);
		const T &operator[](size_t index) const;

		size_t size() const;
		size_t channels() const;
		size_t rows() const;
		size_t cols() const;

		void fill(T val);
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

		static Tensor mult(const Tensor &a, T s);
		static Tensor mult(const Tensor &a, T s, Tensor &y);
		static Tensor mult(const Tensor &a, const Tensor &b);
		static void mult(const Tensor &a, const Tensor &b, Tensor &y);

		static Tensor matMult(const Tensor &a, const Tensor &b);
		static void matMult(const Tensor &a, const Tensor &b, Tensor &y);

		static Tensor transpose(const Tensor &a);
		static void transpose(const Tensor &a, Tensor &y);

		static T sum(const Tensor &a);
		static T sumWith(const Tensor &a, std::function<T(T)> f);

		static Tensor mapWith(const Tensor &a, std::function<T(T)> f);
		static void mapWith(const Tensor &a, std::function<T(T)> f, Tensor &y);

		static void zipWith(const Tensor &a, const Tensor &b, std::function<T(T, T)> f, Tensor &y);

		static void aMinusXMultB(const Tensor &a, const Tensor &b, T x, Tensor &y);

		static void fastSig(const Tensor &a, Tensor &y);
		static void fastSigDeriv(const Tensor &a, Tensor &y);

		static void fastRelu(const Tensor& a, Tensor& y);
		static void fastReluDeriv(const Tensor& a, Tensor& y);

		static void copy(const Tensor &a, Tensor &y);
	};

	template <typename T>
	std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
	{
		os << tensor.str();
		return os;
	}
}