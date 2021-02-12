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

const char* error_string(CUresult retcode) {
  // const char* errmsg;
  // if (cuGetErrorString(retcode, &errmsg) != CUDA_SUCCESS) {
  //   errmsg = "unknown";
  // }
  return "code" + retcode;
  // (void)retcode;
  // return "i have no earthly clue where cuGetErrorString lives";
}

const char* error_string(cudaError_t retcode) {
  return cudaGetErrorString(retcode);
}

const char* error_string(Argus::Status retcode) {
  // FIXME(mdegans)
  return "code " + retcode;
}