#include "MmlSerialization.h"

namespace maxml
{
	BinaryWriter::BinaryWriter(const std::string &path)
		: m_Data(nullptr)
		, m_Size(0)
		, m_Bytes(0)
		, m_Path(path)
	{
		resize(8);
	}

	BinaryWriter::~BinaryWriter()
	{
		std::ofstream f(m_Path, std::ios::binary);
		if (f.is_open())
		{
			f.write(reinterpret_cast<char *>(m_Data), m_Bytes);
		}
		else
		{
			MML_ASSERT(false, "Could not open path '%s' binary writer !", m_Path.c_str());
		}

		delete[] m_Data;
	}

	template<>
	void BinaryWriter::write(const Tensor &value)
	{
		if (m_Data == nullptr)
		{
			MML_ASSERT("Binary writer is not in state to write !");
			return;
		}

		size_t newBytes = m_Bytes + (3 * sizeof(size_t))
			+ (value.size() * sizeof(float));
		if (newBytes >= m_Size)
		{
			resize(newBytes);
		}
		
		*reinterpret_cast<size_t *>(&m_Data[m_Bytes]) = value.channels();
		m_Bytes += sizeof(size_t);

		*reinterpret_cast<size_t *>(&m_Data[m_Bytes]) = value.rows();
		m_Bytes += sizeof(size_t);

		*reinterpret_cast<size_t *>(&m_Data[m_Bytes]) = value.cols();
		m_Bytes += sizeof(size_t);

		std::copy(
			value.data(), value.data() + value.size(),
			reinterpret_cast<float *>(&m_Data[m_Bytes])
		);
		m_Bytes += value.size() * sizeof(float);
	}

	void BinaryWriter::resize(size_t size)
	{
		uint8_t *data = new uint8_t[size];
		MML_ASSERT(data != nullptr, "Failed to allocate memory for binary writer!");
		
		if (m_Data)
		{
			std::copy(m_Data, m_Data + std::min(m_Size, size), data);
			delete[] m_Data;
		}

		m_Size = size;
		m_Data = data;
	}

	BinaryReader::BinaryReader(const std::string &path)
		: m_Data(nullptr)
		, m_Size(0)
		, m_Bytes(0)
	{
		std::ifstream f(path, std::ios::binary);
		if (f.is_open())
		{
			f.seekg(0, std::ios::end);
			size_t size = f.tellg();
			f.seekg(0);

			uint8_t *data = new uint8_t[size];
			MML_ASSERT(data != nullptr, "Failed to allocate memory for binary reader!");
			
			f.read(reinterpret_cast<char *>(data), size);

			m_Data = data;
			m_Size = size;
			m_Bytes = 0;
		}
		else
		{
			MML_ASSERT(false, "Could not open path '%s' binary reader !", path.c_str());
		}
	}

	BinaryReader::~BinaryReader()
	{
		delete[] m_Data;
	}

	template<>
	void BinaryReader::read(Tensor &value)
	{
		if (m_Data == nullptr)
		{
			MML_ASSERT("Binary reader is not in state to read !");
			return;
		}

		size_t newBytes = m_Bytes + (3 * sizeof(size_t))
			+ (value.size() * sizeof(float));
		if (newBytes > m_Size)
		{
			MML_ASSERT(false, "Attempt to read out of buffer for binary reader !");
			return;
		}

		size_t channels = *reinterpret_cast<size_t *>(&m_Data[m_Bytes]);
		m_Bytes += sizeof(size_t);

		size_t rows = *reinterpret_cast<size_t *>(&m_Data[m_Bytes]);
		m_Bytes += sizeof(size_t);

		size_t cols = *reinterpret_cast<size_t *>(&m_Data[m_Bytes]);
		m_Bytes += sizeof(size_t);

		value.resize(channels, rows, cols);

		Tensor::copy(
			value, 
			reinterpret_cast<float *>(&m_Data[m_Bytes]), channels * rows * cols
		);
		m_Bytes += value.size() * sizeof(float);	
	}
}