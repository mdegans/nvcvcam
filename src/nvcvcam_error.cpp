/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "nvcvcam_error.hpp"

namespace nvcvcam {

const char* error_string(CUresult retcode) {
  const char* errmsg;
  if (cuGetErrorString(retcode, &errmsg) != CUDA_SUCCESS) {
    errmsg = "unknown";
  }
  return "code" + retcode;
}

const char* error_string(cudaError_t retcode) {
  return cudaGetErrorString(retcode);
}

}  // namespace nvcvcam
