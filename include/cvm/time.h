#ifndef CVM_TIME_H
#define CVM_TIME_H

#include <chrono>

using cvm_clock = std::chrono::high_resolution_clock;

using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;

#ifdef PROFILE

#define TIME_INIT(id) \
  auto __start_ ## id = cvm_clock::now();

#else
#define TIME_INIT(id)
#endif

#ifdef PROFILE

#define TIME_ELAPSED(id) \
  auto __end_ ## id = cvm_clock::now(); \
  auto __count_ ## id = __end_ ## id - \
    __start_ ## id; \
  std::cout << "Time elapsed: " \
    << (double)(__count_ ## id.count()) / 1000000 \
    << " ms, "

#else
#define TIME_ELAPSED(id)
#endif

#endif // CVM_TIME_H
