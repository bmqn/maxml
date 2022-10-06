#pragma once

#include "maxml/MmlTensor.h"

namespace maxml
{
	class BinaryWriter
	{
	public:
		BinaryWriter(const std::string &path);
		~BinaryWriter();

		template<typename T>
		void write(const T &value)
		{
			static_assert(std::is_trivially_copyable_v<T>, "Type is not trivially copyable !");

			if (m_Data == nullptr)
			{
				MML_ASSERT("Binary writer is not in state to write !");
				return;
			}

			size_t newBytes = m_Bytes + sizeof(T);
			if (newBytes >= m_Size)
			{
				resize(newBytes);
			}

			*reinterpret_cast<T *>(&m_Data[m_Bytes]) = value;
			m_Bytes += sizeof(T);
		}

		template<>
		void write(const Tensor &value);

	private:
		void resize(size_t size);

	private:
		uint8_t *m_Data;
		size_t m_Size;
		size_t m_Bytes;

		std::string m_Path;
	};

	class BinaryReader
	{
	public:
		BinaryReader(const std::string &path);
		~BinaryReader();

		template<typename T>
		void read(T &value)
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
			m_Bytes += sizeof(T);
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

		template<>
		void read(Tensor &value);

	private:
		uint8_t *m_Data;
		size_t m_Size;
		size_t m_Bytes;
	};
}