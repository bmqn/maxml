#pragma once

#include "Common.h"

template<typename T>
struct Tensor
{
	// int n_; // Batches

	int c_; // Channels
	int w_; // Width
	int h_; // Height

	int size_;

	T*  data_;

	Tensor() :
		c_(0), w_(0), h_(0), size_(0), data_(nullptr)
	{
	}

	Tensor(int c, int w, int h) :
		c_(c), w_(w), h_(h), size_(c * w * h), data_(nullptr)
	{
		assert(size_ > 0);

		data_ = new T[size_];

		memset(data_, 0, size_ * sizeof(T));
	}

	Tensor(int c, int w, int h, std::initializer_list<T> data) :
		c_(c), w_(w), h_(h), size_(c* w* h), data_(nullptr)
	{
		assert(size_ > 0);
		assert(data.size() == size_, "Data-shape mismatch");

		data_ = new T[size_];

		std::copy(data.begin(), data.end(), data_);
	}

	~Tensor()
	{
		if (data_) delete[] data_;
	}

	/*Tensor(const Tensor<T>& tensor) :
		c_(tensor.c_), w_(tensor.w_), h_(tensor.h_), size_(tensor.size_), data_(nullptr)
	{
		assert(size_ > 0);

		data_ = new T[size_];

		memcpy(data_, tensor.data_, size_ * sizeof(T));
	}*/

	Tensor(Tensor<T>&& tensor) :
		c_(tensor.c_), w_(tensor.w_), h_(tensor.h_), size_(tensor.size_), data_(tensor.data_)
	{
		tensor.c_ = 0;
		tensor.w_ = 0;
		tensor.h_ = 0;
		tensor.size_ = 0;
		tensor.data_ = nullptr;
	}

	/*Tensor<T>& operator=(const Tensor<T>& tensor)
	{
		if (this == &tensor)
			return *this;

		if (size_ != size_ && data_)
		{
			delete[] data_;
			data_ = new T[tensor.size_];
		}

		c_ = tensor.c_;
		w_ = tensor.w_;
		h_ = tensor.h_;
		size_ = tensor.size_;

		memcpy(data_, tensor.data_, tensor.size_ * sizeof(T));

		return *this;
	}*/

	Tensor<T>& operator=(Tensor<T>&& tensor)
	{
		if (this == &tensor)
			return *this;

		if (data_)
			delete[] data_;

		c_ = tensor.c_;
		w_ = tensor.w_;
		h_ = tensor.h_;
		size_ = tensor.size_;
		data_ = tensor.data_;

		tensor.c_ = 0;
		tensor.w_ = 0;
		tensor.h_ = 0;
		tensor.size_ = 0;
		tensor.data_ = nullptr;

		return *this;
	}

	T& operator() (int c, int w, int h)
	{
		assert(c >= 0 && c < c_);
		assert(w >= 0 && w < w_);
		assert(h >= 0 && h < h_);

		int index = c * (w_ * h_) + w * (h_) + h;
		return data_[index];
	}

	const T& operator() (int c, int w, int h) const
	{
		assert(c >= 0 && c < c_);
		assert(w >= 0 && w < w_);
		assert(h >= 0 && h < h_);

		int index = c * (w_ * h_) + w * (h_) + h;
		return data_[index];
	}

	T& operator[] (int i)
	{
		assert(i >= 0 && i < size_);

		return data_[i];
	}

	const T& operator[] (int i) const
	{
		assert(i >= 0 && i < size_);

		return data_[i];
	}

	void set(T val)
	{
		for (int i = 0; i < size_; i++)
			data_[i] = val;
	}

	std::string str() const
	{
		std::ostringstream ss;

		ss.precision(2);
		ss.fill(' ');

		int maxLen = 0;

		// EXPENSIVE: Finding max number width for alignment...
		for (int i = 0; i < size_; i++)
		{
			std::stringstream tss;
			tss << std::fixed << std::setprecision(2) << data_[i];

			std::string s;
			tss >> s;

			if (s.length() > maxLen)
				maxLen = s.length();
		}

		// Shape
		ss << "shp = (" << c_ << ", " << w_ << ", " << h_ << ")," << std::endl;

		// Array
		ss << "arr = ([";
		for (int c = 0; c < c_; c++)
		{
			if (c > 0) ss << std::setw(9);
			ss << "[";
			for (int w = 0; w < w_; w++)
			{
				if (w > 0) ss << std::setw(10);
				ss << "[";
				for (int h = 0; h < h_; h++)
				{
					int index = c * (w_ * h_) + w * (h_) + h;

					ss << std::fixed << std::right << std::setw(maxLen) << data_[index]; // (int)(data_[index] * 255.0f);
					if (h < h_ - 1) ss << ", ";
				}
				ss << "]";
				if (w < w_ - 1) ss << "," << std::endl;
			}
			ss << "]";
			if (c < c_ - 1) ss << "," << std::endl;
		}
		ss << "])" << std::endl;

		return ss.str();
	}
};

namespace op
{
	template<typename T>
	Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.w_, a.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) + b(c, i, j);
				}

		return std::move(y);
	}

	template<typename T>
	void add(Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) + b(c, i, j);
				}
	}

	template<typename T>
	Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.w_, a.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) - b(c, i, j);
				}

		return std::move(y);
	}

	template<typename T>
	void sub(Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) - b(c, i, j);
				}
	}

	template<typename T>
	Tensor<T> mult(const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.w_, a.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) * b(c, i, j);
				}

		return std::move(y);
	}

	template<typename T>
	void mult(Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) * b(c, i, j);
				}
	}

	template<typename T>
	Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		Tensor<T> y (a.c_, a.w_, b.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					T sum{ 0 };

					for (int k = 0; k < a.h_; k++)
						sum += a(c, i, k) * b(c, k, j);

					y(c, i, j) = sum;
				}

		return std::move(y);
	}

	template<typename T>
	void matmul(Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& b)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					T sum{ 0 };

					for (int k = 0; k < a.h_; k++)
						sum += a(c, i, k) * b(c, k, j);

					y(c, i, j) = sum;
				}
	}

	template<typename T>
	Tensor<T> transpose(const Tensor<T>& a)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.h_, a.w_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, j, i);
				}

		return std::move(y);
	}

	template<typename T>
	void transpose(Tensor<T>& y, const Tensor<T>& a)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, j, i);
				}
	}

	template<typename T>
	T sum(const Tensor<T>& a)
	{
		// TODO: size assertions

		T sum{ 0 };

		for (int c = 0; c < a.c_; c++)
			for (int i = 0; i < a.w_; i++)
				for (int j = 0; j < a.h_; j++)
				{
					sum += a(c, i, j);
				}

		return sum;
	}

	template<typename T>
	void copy(Tensor<T>& y, const Tensor<T>& a)
	{
		// TODO: size assertions

		for (int s = 0; s < a.size_; s++)
			y[s] = a[s];
	}

	template<typename T>
	Tensor<T> resize(const Tensor<T>& a, int c, int w, int h)
	{
		assert(c * w * h == a.size_);

		auto b = Tensor<T>(c, w, h);
		memcpy(b.data_, a.data_, a.size_ * sizeof(T));

		return std::move(b);
	}
};