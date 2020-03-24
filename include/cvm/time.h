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

#endif // CVM_TIME_H
