/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef FBB828EB_207C_4590_B694_0FB4C5CE5C1D
#define FBB828EB_207C_4590_B694_0FB4C5CE5C1D

#include "frame.hpp"
#include "stoppable_thread.hpp"

#include <Argus/Argus.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <nvbuf_utils.h>

#include <opencv2/core/cuda.hpp>

#include <mutex>

namespace nvcvcam {

class Consumer : public thread::StoppableThread {
  Argus::OutputStream* _raw_stream;
  Argus::IEGLOutputStream* _iraw_stream;
  CUeglStreamConnection _cuda_conn;
  cudaStream_t _cuda_stream;
  std::mutex _resource_mtx;
  CUgraphicsResource _raw_resource;

 protected:
  virtual bool setup();
  virtual bool tick();
  virtual bool cleanup();

 public:
  Consumer(Argus::OutputStream* stream)
      : _raw_stream(stream),
        _iraw_stream(nullptr),
        _cuda_conn(nullptr),
        _cuda_stream(nullptr),
        _raw_resource(nullptr){};
  virtual ~Consumer();

  std::unique_ptr<Frame> get_frame();
};

}  // namespace nvcvcam

#endif /* FBB828EB_207C_4590_B694_0FB4C5CE5C1D */
