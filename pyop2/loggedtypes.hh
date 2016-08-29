#ifndef LOGGEDDOUBLE_HH
#define LOGGEDDOUBLE_HH LOGGEDDOUBLE_HH
#include <iostream>

/* ********************************************************************** *
 * Class for logging floating point operations
 *
 * mimics behaviour of double for fundamental operations (+,-,*,/)
 * and records the total number of these operations via the a static class
 * variable totalflops.
 * Arithmetic operations are also overloaded for all datatypes that are
 * compatible with 'double'. E.g. the operation LoggedDouble + int is allowed
 * and returns a LoggedDouble instance
 * ********************************************************************** */
class LoggedDouble {
public: 
  // Constructor
  LoggedDouble() : x(0.0) {}
  template <typename T>
  LoggedDouble(const T x_) : x(x_) {}
  // Return value as double
  const double value() const { return x; }
  // Return total number of FLOPs
  static size_t getTotalFlops() { return totalflops; }
  // Reset total number of FLOPs
  static void resetTotalFlops() { totalflops = 0; }

  // *** Unary operators ***

  // * Same type *
  // Multiply by -1
  LoggedDouble operator-() { totalflops++; return LoggedDouble(-x); }
  // Assignment operator
  LoggedDouble& operator=(LoggedDouble other) { x=other.x; return *this; }

  // *** Binary operators ***
  // += operator
  LoggedDouble& operator+=(LoggedDouble other) { x+=other.x; totalflops++; return *this; }
  LoggedDouble& operator-=(LoggedDouble other) { x-=other.x; totalflops++; return *this; }
  LoggedDouble& operator*=(LoggedDouble other) { x*=other.x; totalflops++; return *this; }
  LoggedDouble& operator/=(LoggedDouble other) { x/=other.x; totalflops++; return *this; }
  // * Other compatible types *
  template <typename T>
  // Assignment operator
  LoggedDouble& operator=(const T y) { x=y; return *this; }
  // += operator
  template <typename T>
  LoggedDouble& operator+=(const T y) { x+=y; totalflops++; return *this; }
  template <typename T>
  LoggedDouble& operator-=(const T y) { x-=y; totalflops++; return *this; }
  template <typename T>
  LoggedDouble& operator*=(const T y) { x*=y; totalflops++; return *this; }
  template <typename T>
  LoggedDouble& operator/=(const T y) { x/=y; totalflops++; return *this; }
private:
  // Internally stored double number
  double x;
  // Total number of FLOPs
  static size_t totalflops;
};

// Initialise total FLOP counter
size_t LoggedDouble::totalflops=0;

// Binary operators
// Output
std::ostream& operator<<(std::ostream& os, const LoggedDouble& a) {
  return os << a.value();
}

// +,-,*,= with same type
LoggedDouble operator+(LoggedDouble lhs, LoggedDouble rhs) { return lhs+=rhs; }
LoggedDouble operator-(LoggedDouble lhs, LoggedDouble rhs) { return lhs-=rhs; }
LoggedDouble operator*(LoggedDouble lhs, LoggedDouble rhs) { return lhs*=rhs; }
LoggedDouble operator/(LoggedDouble lhs, LoggedDouble rhs) { return lhs/=rhs; }

// +,-,*,= with other compatible type
template <typename T>
LoggedDouble operator+(LoggedDouble lhs, const T rhs) { return lhs+=rhs; }
template <typename T>
LoggedDouble operator-(LoggedDouble lhs, const T rhs) { return lhs-=rhs; }
template <typename T>
LoggedDouble operator*(LoggedDouble lhs, const T rhs) { return lhs*=rhs; }
template <typename T>
LoggedDouble operator/(LoggedDouble lhs, const T rhs) { return lhs/=rhs; }
template <typename T>
LoggedDouble operator+(const T lhs, LoggedDouble rhs) { return rhs+=lhs; }
template <typename T>
LoggedDouble operator-(const T lhs, LoggedDouble rhs) { return rhs-=lhs; }
template <typename T>
LoggedDouble operator*(const T lhs, LoggedDouble rhs) { return rhs*=lhs; }
template <typename T>
LoggedDouble operator/(const T lhs, LoggedDouble rhs) {
  LoggedDouble tmp(lhs);
  tmp /= rhs.value();
  return tmp;
}

// Comparisons
bool operator==(LoggedDouble lhs, LoggedDouble rhs) { return lhs.value() == rhs.value(); }
bool operator!=(LoggedDouble lhs, LoggedDouble rhs) { return lhs.value() != rhs.value(); }
bool operator<(LoggedDouble lhs, LoggedDouble rhs) { return lhs.value() < rhs.value(); }
bool operator>(LoggedDouble lhs, LoggedDouble rhs) { return lhs.value() > rhs.value(); }
bool operator<=(LoggedDouble lhs, LoggedDouble rhs) { return lhs.value() <= rhs.value(); }
bool operator>=(LoggedDouble lhs, LoggedDouble rhs) { return lhs.value() >= rhs.value(); }

template <typename T>
bool operator==(LoggedDouble lhs, const T rhs) { return lhs.value() == rhs; }
template <typename T>
bool operator!=(LoggedDouble lhs, const T rhs) { return lhs.value() != rhs; }
template <typename T>
bool operator<(LoggedDouble lhs, const T rhs) { return lhs.value() < rhs; }
template <typename T>
bool operator>(LoggedDouble lhs, const T rhs) { return lhs.value() > rhs; }
template <typename T>
bool operator<=(LoggedDouble lhs, const T rhs) { return lhs.value() <= rhs; }
template <typename T>
bool operator>=(LoggedDouble lhs, const T rhs) { return lhs.value() >= rhs; }

template <typename T>
bool operator==(const T lhs, LoggedDouble rhs) { return lhs == rhs.value(); }
template <typename T>
bool operator!=(const T lhs, LoggedDouble rhs) { return lhs != rhs.value(); }
template <typename T>
bool operator<(const T lhs, LoggedDouble rhs) { return lhs < rhs.value(); }
template <typename T>
bool operator>(const T lhs, LoggedDouble rhs) { return lhs > rhs.value(); }
template <typename T>
bool operator<=(const T lhs, LoggedDouble rhs) { return lhs <= rhs.value(); }
template <typename T>
bool operator>=(const T lhs, LoggedDouble rhs) { return lhs >= rhs.value(); }

// absolute value function
LoggedDouble fabs(LoggedDouble tmp) { return LoggedDouble(fabs(tmp.value())); }
#endif // LOGGEDDOUBLE
