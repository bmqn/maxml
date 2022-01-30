#pragma once

#include <string>
#include <functional>

namespace maxml
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
		size_t m_Channels;
		size_t m_Rows;
		size_t m_Cols;
		size_t m_Size;

		T* m_Data;

		// TODO: Is this a good way to handle ownership ?
		//       Perhaps we should have a TensorView instead...
		bool m_Owned;
	
	private:
		Tensor(size_t chnls, size_t rows, size_t cols, T* data);

	public:
		Tensor();
		Tensor(size_t chnls, size_t rows, size_t cols);
		Tensor(std::initializer_list<T> data);
		Tensor(std::initializer_list<std::initializer_list<T>> data);
		Tensor(const Tensor<T>& tensor);
		Tensor(Tensor<T>&& tensor) noexcept;
		
		~Tensor();

		Tensor<T>& operator=(const Tensor<T>& tensor);
		Tensor<T>& operator=(Tensor<T>&& tensor) noexcept;

		Tensor<T> operator()(size_t chnl) const;

		T& operator()(size_t chnl, size_t row, size_t col);
		const T& operator()(size_t chnl, size_t row, size_t col) const;

		T& operator[](size_t index);
		const T& operator[](size_t index) const;

		size_t size() const;
		size_t channels() const;
		size_t rows() const;
		size_t cols() const;

		void fill(T val);
		void fill(size_t chnl, Tensor<T>& val);

		void resize(size_t c, size_t w, size_t h);

		std::string str() const;
	
	public:
		static Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);
		static void add(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y);
		
		static Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b);
		static void sub(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y);
		
		static Tensor<T> mult(const Tensor<T>& a, T s);
		static Tensor<T> mult(const Tensor<T>& a, T s, Tensor<T>& y);
		static Tensor<T> mult(const Tensor<T>& a, const Tensor<T>& b);
		static void mult(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y);

		static Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);
		static void matmul(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& y, bool transposeA = false, bool transposeB = false);

		static Tensor<T> transpose(const Tensor<T>& a);
		static void transpose(const Tensor<T>& a, Tensor<T>& y);

		static T sum(const Tensor<T>& a);
		static T sumWith(const Tensor<T>& a, std::function<T(T)> f);
		
		static Tensor<T> map(const Tensor<T>& a, std::function<T(T)> f);
		static void map(const Tensor<T>& a, std::function<T(T)> f, Tensor<T>& y);
		
		static void zip(const Tensor<T>& a, const Tensor<T>& b, std::function<T(T, T)> f, Tensor<T>& y);
		
		static void copy(const Tensor<T>& a, Tensor<T>& y);
	};
}