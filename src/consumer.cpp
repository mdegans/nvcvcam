/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "consumer.hpp"

#include "nvcvcam_error.hpp"

namespace nvcvcam {

bool Consumer::setup() {
  Argus::Status err;
  CUresult cu_err;
  cudaError_t cu_err_b;

  DEBUG << "consumer:Starting up.";

  cu_err_b = cudaStreamCreate(&_cuda_stream);
  if (cu_err_b) {
    ERROR << "consumer:Could not create cuda stream because: "
          << error_string(cu_err_b) << ".";
    return false;
  }

  if (!_raw_stream) {
    ERROR << "consumer:Got null stream.";
    return false;
  }

  _iraw_stream = Argus::interface_cast<Argus::IEGLOutputStream>(_raw_stream);
  if (!_iraw_stream) {
    ERROR << "consumer:Could not get IEGLOutputStream interface.";
    return false;
  }

  err = _iraw_stream->waitUntilConnected();
  if (err) {
    ERROR << "consumer:Could not connect to IEGLOutputStream (status" << err
          << ").";
    return false;
  }

  cu_err =
      cuEGLStreamConsumerConnect(&_cuda_conn, _iraw_stream->getEGLStream());
  if (cu_err) {
    ERROR << "consumer:Could not connect CUDA to OutputStream because: "
          << error_string(cu_err) << ".";
    return false;
  }

  return true;
}

bool Consumer::tick() {
  CUresult cu_err;
  CUgraphicsResource tmp_res = nullptr;

  DEBUG << "consumer:Getting an image from the EGLStream.";
  if (auto cu_err = cuEGLStreamConsumerAcquireFrame(&_cuda_conn, &tmp_res,
                                                    &_cuda_stream, -1)) {
    ERROR << "consumer:Could not get image from the EglStream becuase: "
          << error_string(cu_err) << ".";
    return false;
  }

  std::unique_lock<std::mutex> lock(_resource_mtx);
  if (_raw_resource) {
    DEBUG << "consumer:Scheduling frame release.";
    cuEGLStreamConsumerReleaseFrame(&_cuda_conn, _raw_resource, &_cuda_stream);
  }
  _raw_resource = tmp_res;
  return true;
}

std::unique_ptr<Frame> Consumer::get_frame() {
  if (!ready()) {
    ERROR << "consumer:Not ready.";
    return nullptr;
  }
  DEBUG << "consumer:Waiting for resource lock.";
  std::unique_lock<std::mutex> lock(_resource_mtx);
  DEBUG << "consumer:Creating Frame.";
  auto frame = std::make_unique<Frame>(_raw_resource, _cuda_conn, _cuda_stream);
  // frame now owns _raw_resource and will release it
  _raw_resource = nullptr;
  return frame;
}

bool Consumer::cleanup() {
  DEBUG << "consumer:Disconnecting cuEGLStream.";
  if (_cuda_conn) {
    cuEGLStreamConsumerDisconnect(&_cuda_conn);
    _cuda_conn = nullptr;
  }

  DEBUG << "consumer:Disconnecting EGLStream.";
  if (_iraw_stream) {
    _iraw_stream->disconnect();
  }
  _iraw_stream = nullptr;
  _raw_stream = nullptr;

  std::unique_lock<std::mutex> lock(_resource_mtx);
  if (_raw_resource) {
    DEBUG << "consumer:Scheduling frame release.";
    cuEGLStreamConsumerReleaseFrame(&_cuda_conn, _raw_resource, &_cuda_stream);
  }
  lock.release();

  if (auto err = cudaStreamSynchronize(_cuda_stream)) {
    ERROR << "consumer:Failed to sync cuda stream because: "
          << error_string(err) << ".";
  }
  if (auto err = cudaStreamDestroy(_cuda_stream)) {
    ERROR << "consumer:Failed to destroy cuda stream because: "
          << error_string(err) << ".";
  }

  _cuda_stream = nullptr;

  return true;
}

Consumer::~Consumer() {
  DEBUG << "consumer:Destructor reached.";
}

}  // namespace nvcvcam
