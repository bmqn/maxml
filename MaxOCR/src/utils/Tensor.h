#pragma once

#include <vector>
#include <array>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <assert.h>

template<typename T>
struct Tensor
{
	int n_; // Batches
	int c_; // Channels
	int w_; // Width
	int h_; // Height
	int size_;
	T* data_;

	Tensor(int c, int w, int h)
		: c_(c), w_(w), h_(h), size_(c * w * h), data_(nullptr)
	{
		assert(size_ > 0);

		data_ = new T[size_];

		memset(data_, 0, size_ * sizeof(T));
	}

	Tensor(const Tensor<T>& tensor) = delete;

	Tensor(Tensor<T>&& tensor)
		: c_(tensor.c_), w_(tensor.w_), h_(tensor.h_), size_(tensor.size_), data_(tensor.data_)
	{
		tensor.c_ = 0;
		tensor.w_ = 0;
		tensor.h_ = 0;
		tensor.size_ = 0;
		tensor.data_ = nullptr;
	}

	~Tensor()
	{
		if (data_) delete[] data_;
	}

	Tensor<T>& operator=(const Tensor<T>& tensor) = delete;

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

	void setTo(T val)
	{
		for (int i = 0; i < size_; i++)
			data_[i] = val;
	}

	friend std::ostream& operator<< (std::ostream& out, const Tensor& tensor)
	{
		out << tensor.str();
		return out;
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