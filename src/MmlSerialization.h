#pragma once

#include "maxml/MmlTensor.h"

namespace maxml
{
	class BinaryWriter;
	class BinaryReader;

	namespace BinaryWriterImpl
	{
		template<typename T>
		void write(BinaryWriter &writer, const T &value);

		template<>
		void write(BinaryWriter &writer, const Tensor &value);
	}

	class BinaryWriter
	{
		template<typename T>
		friend void BinaryWriterImpl::write(BinaryWriter &writer, const T &value);

	public:
		BinaryWriter(const std::string &path);
		~BinaryWriter();

		template<typename T>
		void write(const T &value)
		{
			BinaryWriterImpl::write<T>(*this, value);
		}

	private:
		void resize(size_t size);

	private:
		uint8_t *m_Data;
		size_t m_Size;
		size_t m_Bytes;

		std::string m_Path;
	};

	namespace BinaryWriterImpl
	{
		template<typename T>
		void write(BinaryWriter &writer, const T &value)
		{
			static_assert(std::is_trivially_copyable_v<T>, "Type is not trivially copyable !");

			if (writer.m_Data == nullptr)
			{
				MML_ASSERT("Binary writer is not in state to write !");
				return;
			}

			size_t newBytes = writer.m_Bytes + sizeof(T);
			if (newBytes >= writer.m_Size)
			{
				writer.resize(newBytes);
			}

			*reinterpret_cast<T *>(&writer.m_Data[writer.m_Bytes]) = value;
			writer.m_Bytes += sizeof(T);
		}
	}

	namespace BinaryReaderImpl
	{
		template<typename T>
		void read(BinaryReader &reader, T &value);
		
		template<>
		void read(BinaryReader &reader, Tensor &value);
	}

	class BinaryReader
	{
		template<typename T>
		friend void BinaryReaderImpl::read(BinaryReader &reader, T &value);

	public:
		BinaryReader(const std::string &path);
		~BinaryReader();

		template<typename T>
		void read(T &value)
		{
			BinaryReaderImpl::read<T>(*this, value);
		}

		template<typename T>
		void peek(T &value)
		{
			static_assert(std::is_trivially_copyable_v<T>, "Type is not trivially copyable !");

			if (m_Data == nullptr)
			{
				MML_ASSERT("Binary reader is not in state to read !");
				return;
			}

			size_t newBytes = m_Bytes + sizeof(T);
			if (newBytes > m_Size)
			{
				MML_ASSERT(false, "Attempt to read out of buffer for binary reader !");
				return;
			}

			value = *reinterpret_cast<T *>(&m_Data[m_Bytes]);
		}

	private:
		uint8_t *m_Data;
		size_t m_Size;
		size_t m_Bytes;
	};


	namespace BinaryReaderImpl
	{
		template<typename T>
		void read(BinaryReader &reader, T &value)
		{
			static_assert(std::is_trivially_copyable_v<T>, "Type is not trivially copyable !");

			if (reader.m_Data == nullptr)
			{
				MML_ASSERT("Binary reader is not in state to read !");
				return;
			}

			size_t newBytes = reader.m_Bytes + sizeof(T);
			if (newBytes > reader.m_Size)
			{
				MML_ASSERT(false, "Attempt to read out of buffer for binary reader !");
				return;
			}

			value = *reinterpret_cast<T *>(&reader.m_Data[reader.m_Bytes]);
			reader.m_Bytes += sizeof(T);
		}
	}
}