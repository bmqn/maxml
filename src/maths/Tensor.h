#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>

#include <assert.h>

#include <iostream>

namespace mocr
{

	template <typename T>
	struct Tensor
	{
		// int n_; // Batches

		int c_; // Channels
		int w_; // Width
		int h_; // Height

		int size_;

		T *data_;

		Tensor(int c, int w, int h) : c_(c), w_(w), h_(h), size_(c * w * h), data_(nullptr)
		{
			std::cout << "Creating tensor!" << std::endl;

			assert(size_ > 0);

			data_ = new T[size_];

			std::fill(data_, data_ + size_, 0);
		}

		Tensor(int c, int w, int h, std::initializer_list<T> data) : c_(c), w_(w), h_(h), size_(c * w * h), data_(nullptr)
		{
			std::cout << "Creating tensor!" << std::endl;

			assert(size_ > 0);
			assert(data.size() == size_);

			data_ = new T[size_];

			std::copy(data.begin(), data.end(), data_);
		}

		Tensor(std::initializer_list<std::initializer_list<T>> data)
		{
			std::cout << "Creating tensor!" << std::endl;

			c_ = 1;
			w_ = data.size();
			h_ = 0;

			if (w_ > 0)
				h_ = data.begin()->size();

			size_ = w_ * h_;

			assert(size_ > 0);

			data_ = new T[size_];

			int i = 0;
			for (auto &row : data)
			{
				std::copy(row.begin(), row.end(), data_ + h_ * i);
				i++;
			}
		}

		~Tensor()
		{
			delete[] data_;

			std::cout << "Deleting tensor!" << std::endl;
		}

		Tensor(const Tensor<T> &tensor) : c_(tensor.c_), w_(tensor.w_), h_(tensor.h_), size_(tensor.size_), data_(nullptr)
		{
			assert(size_ > 0);

			data_ = new T[size_];

			std::copy(tensor.data_, tensor.data_ + size_, data_);

			std::cout << "Copying tensor!" << std::endl;
		}

		Tensor(Tensor<T> &&tensor) : c_(tensor.c_), w_(tensor.w_), h_(tensor.h_), size_(tensor.size_), data_(tensor.data_)
		{
			tensor.c_ = 0;
			tensor.w_ = 0;
			tensor.h_ = 0;
			tensor.size_ = 0;
			tensor.data_ = nullptr;

			std::cout << "Moving tensor!" << std::endl;
		}

		Tensor<T> &operator=(const Tensor<T> &tensor)
		{
			if (this == &tensor)
				return *this;

			if (data_ && size_ != tensor.size_)
			{
				delete[] data_;
				data_ = new T[tensor.size_];
			}

			c_ = tensor.c_;
			w_ = tensor.w_;
			h_ = tensor.h_;
			size_ = tensor.size_;

			std::copy(tensor.data_, tensor.data_ + size_, data_);

			std::cout << "Assigning (copying) tensor!" << std::endl;

			return *this;
		}

		Tensor<T> &operator=(Tensor<T> &&tensor)
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

			std::cout << "Assigning (moving) tensor!" << std::endl;

			return *this;
		}

		T &operator()(int c, int w, int h)
		{
			assert(c >= 0 && c < c_);
			assert(w >= 0 && w < w_);
			assert(h >= 0 && h < h_);

			int index = c * (w_ * h_) + w * (h_) + h;
			return data_[index];
		}

		const T &operator()(int c, int w, int h) const
		{
			assert(c >= 0 && c < c_);
			assert(w >= 0 && w < w_);
			assert(h >= 0 && h < h_);

			int index = c * (w_ * h_) + w * (h_) + h;
			return data_[index];
		}

		T &operator[](int i)
		{
			assert(i >= 0 && i < size_);

			return data_[i];
		}

		const T &operator[](int i) const
		{
			assert(i >= 0 && i < size_);

			return data_[i];
		}

		void fill(T val)
		{
			for (int i = 0; i < size_; i++)
				data_[i] = val;
		}

		void fill(int c, Tensor<T> &val)
		{
			assert(c >= 0 && c < c_ && val.w_ == w_ && val.h_ == val.h_);

			for (int i = 0; i < w_; i++)
				for (int j = 0; j < h_; j++)
					this->operator()(c, i, j) = val(c, i, j);
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
				if (c > 0)
					ss << std::setw(9);
				ss << "[";
				for (int w = 0; w < w_; w++)
				{
					if (w > 0)
						ss << std::setw(10);
					ss << "[";
					for (int h = 0; h < h_; h++)
					{
						int index = c * (w_ * h_) + w * (h_) + h;

						ss << std::fixed << std::right << std::setw(maxLen) << data_[index]; // (int)(data_[index] * 255.0f);
						if (h < h_ - 1)
							ss << ", ";
					}
					ss << "]";
					if (w < w_ - 1)
						ss << "," << std::endl;
				}
				ss << "]";
				if (c < c_ - 1)
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

		Tensor<T> y(a.c_, a.w_, a.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) + b(c, i, j);
				}

		return y;
	}

	template <typename T>
	void add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) + b(c, i, j);
				}
	}

	template <typename T>
	Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.w_, a.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) - b(c, i, j);
				}

		return y;
	}

	template <typename T>
	void sub(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) - b(c, i, j);
				}
	}

	template <typename T>
	Tensor<T> mult(const Tensor<T> &a, const Tensor<T> &b)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.w_, a.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, i, j) * b(c, i, j);
				}

		return y;
	}

	template <typename T>
	Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b)
	{
		assert(a.c_ == b.c_ && a.w_ == b.h_ && a.h_ == b.w_);

		Tensor<T> y(a.c_, a.w_, b.h_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					T sum{0};

					for (int k = 0; k < a.h_; k++)
						sum += a(c, i, k) * b(c, k, j);

					y(c, i, j) = sum;
				}

		return y;
	}

	template <typename T>
	void matmul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					T sum{0};

					for (int k = 0; k < a.h_; k++)
						sum += a(c, i, k) * b(c, k, j);

					y(c, i, j) = sum;
				}
	}

	template <typename T>
	Tensor<T> transpose(const Tensor<T> &a)
	{
		// TODO: size assertions

		Tensor<T> y(a.c_, a.h_, a.w_);

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, j, i);
				}

		return y;
	}

	template <typename T>
	void transpose(const Tensor<T> &a, Tensor<T> &y)
	{
		// TODO: size assertions

		for (int c = 0; c < y.c_; c++)
			for (int i = 0; i < y.w_; i++)
				for (int j = 0; j < y.h_; j++)
				{
					y(c, i, j) = a(c, j, i);
				}
	}

	template <typename T>
	T sum(const Tensor<T> &a)
	{
		// TODO: size assertions

		T sum{0};

		for (int c = 0; c < a.c_; c++)
			for (int i = 0; i < a.w_; i++)
				for (int j = 0; j < a.h_; j++)
				{
					sum += a(c, i, j);
				}

		return sum;
	}

	template <typename T>
	Tensor<T> resize(const Tensor<T> &a, int c, int w, int h)
	{
		assert(c * w * h == a.size_);

		Tensor<T> y(a);

		y.c_ = c;
		y.w_ = w;
		y.h_ = h;

		return y;
	}
}