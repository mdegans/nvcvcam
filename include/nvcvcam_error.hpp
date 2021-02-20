#ifndef EF3D5456_D6DE_4825_9D87_B3655837B3D3
#define EF3D5456_D6DE_4825_9D87_B3655837B3D3

#include <Argus/Argus.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/log/trivial.hpp>

#include <memory>
#include <stdexcept>
#include <string>

// https://stackoverflow.com/a/26221725/11049585
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  int size = snprintf(nullptr, 0, format.c_str(), args...) +
             1;  // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1);  // We don't want the '\0' inside
}

#define ERROR BOOST_LOG_TRIVIAL(error)
#define INFO BOOST_LOG_TRIVIAL(info)
#define DEBUG BOOST_LOG_TRIVIAL(debug)
#define WARNING BOOST_LOG_TRIVIAL(warning)

/**
 * Report and return an error that was first detected in some method
 * called by the current method.
 */
#define PROPAGATE_ERROR(_err)   \
  do {                          \
    bool peResult = (_err);     \
    if (peResult != true) {     \
      ERROR << "(propagating)"; \
      return false;             \
    }                           \
  } while (0)

/**
 * Report and return an error that was first detected in the current method.
 */
#define ORIGINATE_ERROR(_str, ...)                 \
  do {                                             \
    ERROR << string_format((_str), ##__VA_ARGS__); \
    return false;                                  \
  } while (0)

#define CUDA_OK(ret) (ret != cudaSuccess)

namespace nvcvcam {

const char* error_string(CUresult retcode);

const char* error_string(cudaError_t retcode);

}  // namespace nvcvcam

#endif /* EF3D5456_D6DE_4825_9D87_B3655837B3D3 */
