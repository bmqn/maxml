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
			MML_ASSERT(false, "Binary writer could not open path '%s' !", m_Path.c_str());
		}

		delete[] m_Data;
	}

	namespace BinaryWriterImpl
	{
		template<>
		void write(BinaryWriter &writer, const Tensor &value)
		{
			if (writer.m_Data == nullptr)
			{
				MML_ASSERT("Binary writer is not in state to write !");
				return;
			}

			size_t newBytes = writer.m_Bytes + (3 * sizeof(size_t))
				+ (value.size() * sizeof(float));
			if (newBytes >= writer.m_Size)
			{
				writer.resize(newBytes);
			}
			
			*reinterpret_cast<size_t *>(&writer.m_Data[writer.m_Bytes]) = value.channels();
			writer.m_Bytes += sizeof(size_t);

			*reinterpret_cast<size_t *>(&writer.m_Data[writer.m_Bytes]) = value.rows();
			writer.m_Bytes += sizeof(size_t);

			*reinterpret_cast<size_t *>(&writer.m_Data[writer.m_Bytes]) = value.cols();
			writer.m_Bytes += sizeof(size_t);

			Tensor::copy(
				reinterpret_cast<float *>(&writer.m_Data[writer.m_Bytes]), value.size(),
				value
			);
			writer.m_Bytes += value.size() * sizeof(float);
		}
	}

	void BinaryWriter::resize(size_t size)
	{
		uint8_t *data = new uint8_t[size];
		MML_ASSERT(data != nullptr, "Binary writer failed to allocate memory !");
		
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
			MML_ASSERT(data != nullptr, "Binary reader failed to allocate memory !");
			
			f.read(reinterpret_cast<char *>(data), size);

			m_Data = data;
			m_Size = size;
			m_Bytes = 0;
		}
		else
		{
			MML_ASSERT(false, "Binary reader could not open path '%s' !", path.c_str());
		}
	}

	BinaryReader::~BinaryReader()
	{
		delete[] m_Data;
	}

	namespace BinaryReaderImpl
	{
		template<>
		void read(BinaryReader &reader, Tensor &value)
		{
			if (reader.m_Data == nullptr)
			{
				MML_ASSERT("Binary reader is not in state to read !");
				return;
			}

			size_t newBytes = reader.m_Bytes + (3 * sizeof(size_t))
				+ (value.size() * sizeof(float));
			if (newBytes > reader.m_Size)
			{
				MML_ASSERT(false, "Binary reader attempted to read out of buffer !");
				return;
			}

			size_t channels = *reinterpret_cast<size_t *>(&reader.m_Data[reader.m_Bytes]);
			reader.m_Bytes += sizeof(size_t);

			size_t rows = *reinterpret_cast<size_t *>(&reader.m_Data[reader.m_Bytes]);
			reader.m_Bytes += sizeof(size_t);

			size_t cols = *reinterpret_cast<size_t *>(&reader.m_Data[reader.m_Bytes]);
			reader.m_Bytes += sizeof(size_t);

			if (channels * rows * cols <= 0)
			{
				MML_ASSERT(false, "Binary reader read invalid tensor size !");
				reader.m_Bytes -= 3 * sizeof(size_t);
				return;
			}

			value.resize(channels, rows, cols);
			Tensor::copy(
				value, 
				reinterpret_cast<float *>(&reader.m_Data[reader.m_Bytes]), channels * rows * cols
			);
			reader.m_Bytes += value.size() * sizeof(float);
		}
	}
}