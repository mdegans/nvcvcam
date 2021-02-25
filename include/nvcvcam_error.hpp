#ifndef EF3D5456_D6DE_4825_9D87_B3655837B3D3
#define EF3D5456_D6DE_4825_9D87_B3655837B3D3

#include <Argus/Argus.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <boost/log/trivial.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#define ERROR BOOST_LOG_TRIVIAL(error)
#define INFO BOOST_LOG_TRIVIAL(info)
#define DEBUG BOOST_LOG_TRIVIAL(debug)
#define WARNING BOOST_LOG_TRIVIAL(warning)

#define CUDA_OK(ret) (ret != cudaSuccess)

namespace nvcvcam {

const char* error_string(CUresult retcode);

const char* error_string(cudaError_t retcode);

const char* error_string(NppStatus status);

}  // namespace nvcvcam

#endif /* EF3D5456_D6DE_4825_9D87_B3655837B3D3 */
