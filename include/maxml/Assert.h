#pragma once

#ifndef NDEBUG
#include <cassert>
#define MOCR_ASSERT(condition)                                            \
do                                                                        \
{                                                                         \
  if (!(condition))                                                       \
  {                                                                       \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__; \
      std::cerr << " inside " << __FUNCTION__ << std::endl;               \
      abort();                                                            \
  }                                                                       \
} while(0)
#else
#define MOCR_ASSERT(condition) \
do                             \
{                              \
} while(0)
#endif