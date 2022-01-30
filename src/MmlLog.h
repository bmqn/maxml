#pragma once

#include "MmlConfig.h"

#include <iostream>

namespace maxml
{
	class MmlLog
	{
	public:
		template<typename ...Args>
		static void LogInfo(const char* file, unsigned long line, const char* format, Args... args)
		{
			fprintf(stdout, "[INFO] %s(%lu): ", file, line);
			fprintf(stdout, format, args...);
			fprintf(stdout, "\n");
		}

		template<typename ...Args>
		static void LogError(const char* file, unsigned long line, const char* format, Args... args)
		{
			fprintf(stderr, "[ERROR] %s(%lu): ", file, line);
			fprintf(stderr, format, args...);
			fprintf(stderr, "\n");
		}
	};
}

#define _MML_EXPAND(x) x
#define _MML_VARGS(_9, _8, _7, _6, _5, _4, _3, _2, _1, N, ...) N

// ----- Logging -----
#if MML_LOGGING

#define _MML_LOGI1(format)      MmlLog::LogInfo(__FILE__, __LINE__, format);
#define _MML_LOGI2(format, ...) MmlLog::LogInfo(__FILE__, __LINE__, format, __VA_ARGS__);

#define _MML_LOGI_CHOOSER(...) _MML_EXPAND( \
_MML_VARGS(__VA_ARGS__,                     \
_MML_LOGI2, _MML_LOGI2, _MML_LOGI2,         \
_MML_LOGI2, _MML_LOGI2, _MML_LOGI2,         \
_MML_LOGI2, _MML_LOGI2, _MML_LOGI1)         \
)

#define MML_LOGI(...) _MML_EXPAND(_MML_LOGI_CHOOSER(__VA_ARGS__)(__VA_ARGS__))

#else

#define MML_LOGI(...)

#endif // MML_LOGGING

// --- Assertions ---
#if MML_ASSERTION

#define _MML_ASSERT1(condition)                          \
do                                                       \
{                                                        \
	if (!(condition))                                    \
	{                                                    \
		MmlLog::LogError(__FILE__, __LINE__, "Assertion failed!"); \
	}                                                    \
} while(0)

#define _MML_ASSERT2(condition, format)                          \
do                                                               \
{                                                                \
	if (!(condition))                                            \
	{                                                            \
		MmlLog::LogError(__FILE__, __LINE__, "Assertion failed! " ## format); \
	}                                                            \
} while (0)

#define _MML_ASSERT3(condition, format, ...)                                  \
do                                                                            \
{                                                                             \
	if (!(condition))                                                         \
	{                                                                         \
		MmlLog::LogError(__FILE__, __LINE__, "Assertion failed! " ## format, __VA_ARGS__); \
	}                                                                         \
} while (0)

#define _MML_ASSERT_CHOOSER(...) _MML_EXPAND( \
_MML_VARGS(__VA_ARGS__,                       \
_MML_ASSERT3, _MML_ASSERT3, _MML_ASSERT3,     \
_MML_ASSERT3, _MML_ASSERT3, _MML_ASSERT3,     \
_MML_ASSERT3, _MML_ASSERT2, _MML_ASSERT1)     \
)

#define MML_ASSERT(...) _MML_EXPAND(_MML_ASSERT_CHOOSER(__VA_ARGS__)(__VA_ARGS__))

#else

#define MML_ASSERT(...)

#endif  // MML_ASSERTION