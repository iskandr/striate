#ifndef PYASSERT_H
#define PYASSERT_H

#ifdef assert
#undef assert
#endif

#include <string>
#include <stdarg.h>

inline std::string VStringPrintf(const char* fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt, l);
  return std::string(buffer);
}

inline std::string StringPrintf(const char* fmt, ...) {
  va_list l;
  va_start(l, fmt);
  std::string result = VStringPrintf(fmt, l);
  va_end(l);

  return result;
}

struct Exception {
  std::string why_;
  std::string file_;
  int line_;
  Exception(std::string c, const char* file, int line) : why_(c), file_(file), line_(line) {}
};

#define assert(expr) \
    do {\
    if (!(expr)) {\
        throw Exception(#expr, __FILE__, __LINE__);\
    }\
    } while(0)

#endif
