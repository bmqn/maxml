#pragma once

#include "MmlConfig.h"

#define _MML_EXPAND(x) x
#define _MML_VARGS(_9, _8, _7, _6, _5, _4, _3, _2, _1, N, ...) N

//-------------------------------------------------------------------------------------------------
//	Logging
//-------------------------------------------------------------------------------------------------
#if MML_LOGGING

#define _MML_LOGI1(format)                                                                        \
	do                                                                                            \
	{                                                                                             \
		fprintf(stdout, "[INFO] %s:%d ", __FILE__, __LINE__);                                     \
		fprintf(stdout, format);                                                                  \
		fprintf(stdout, "\n");                                                                    \
	} while (0)

#define _MML_LOGI2(format, ...)                                                                   \
	do                                                                                            \
	{                                                                                             \
		fprintf(stdout, "[INFO] %s:%d ", __FILE__, __LINE__);                                     \
		fprintf(stdout, format, __VA_ARGS__);                                                     \
		fprintf(stdout, "\n");                                                                    \
	} while (0)

#define _MML_LOGI_CHOOSER(...) _MML_EXPAND(_MML_VARGS(__VA_ARGS__,                                \
	_MML_LOGI2, _MML_LOGI2, _MML_LOGI2,                                                           \
	_MML_LOGI2, _MML_LOGI2, _MML_LOGI2,                                                           \
	_MML_LOGI2, _MML_LOGI2, _MML_LOGI1))

#define MML_LOG(...) _MML_EXPAND(_MML_LOGI_CHOOSER(__VA_ARGS__)(__VA_ARGS__))

#else

#define MML_LOG(...) \
do                    \
{                     \
} while (0)

#endif // MML_LOGGING

//-------------------------------------------------------------------------------------------------
//	Assertions
//-------------------------------------------------------------------------------------------------
#if MML_ASSERTION

#define _MML_ASSERT1(condition)                                                                   \
	do                                                                                            \
	{                                                                                             \
		if (!(condition))                                                                         \
		{                                                                                         \
			fprintf(stdout, "[ERROR] %s:%d ", __FILE__, __LINE__);                                \
			fprintf(stdout, "Assertion failed!");                                                 \
			fprintf(stdout, "\n");                                                                \
		}                                                                                         \
	} while (0)

#define _MML_ASSERT2(condition, format)                                                           \
	do                                                                                            \
	{                                                                                             \
		if (!(condition))                                                                         \
		{                                                                                         \
			fprintf(stdout, "[ERROR] %s:%d ", __FILE__, __LINE__);                                \
			fprintf(stdout, "Assertion failed! ");                                                \
			fprintf(stdout, format);                                                              \
			fprintf(stdout, "\n");                                                                \
		}                                                                                         \
	} while (0)

#define _MML_ASSERT3(condition, format, ...)                                                      \
	do                                                                                            \
	{                                                                                             \
		if (!(condition))                                                                         \
		{                                                                                         \
			fprintf(stdout, "[ERROR] %s:%d ", __FILE__, __LINE__);                                \
			fprintf(stdout, "Assertion failed! ");                                                \
			fprintf(stdout, format, __VA_ARGS__);                                                 \
			fprintf(stdout, "\n");                                                                \
		}                                                                                         \
	} while (0)

#define _MML_ASSERT_CHOOSER(...) _MML_EXPAND(_MML_VARGS(__VA_ARGS__,                              \
	_MML_ASSERT3, _MML_ASSERT3, _MML_ASSERT3,                                                     \
	_MML_ASSERT3, _MML_ASSERT3, _MML_ASSERT3,                                                     \
	_MML_ASSERT3, _MML_ASSERT2, _MML_ASSERT1))

#define MML_ASSERT(...) _MML_EXPAND(_MML_ASSERT_CHOOSER(__VA_ARGS__)(__VA_ARGS__))

#else

#define MML_ASSERT(...)                                                                           \
do                                                                                                \
{                                                                                                 \
} while (0)

#endif // MML_ASSERTION