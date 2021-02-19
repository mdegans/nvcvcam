/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "consumer.hpp"

#include "nvcvcam_error.hpp"

namespace nvcvcam {

bool Consumer::setup() {
  Argus::Status err;
  CUresult cu_err;
  cudaError_t cu_err_b;

  cu_err_b = cudaStreamCreate(&_cuda_stream);
  if (cu_err_b) {
    ERROR << "consumer:Could not create cuda stream (cudaError_t " << cu_err_b
          << ").";
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

  if (auto ret = cudaStreamSynchronize(_cuda_stream)) {
    ERROR << "consumer:Failed to sync cuda stream. (code: " << ret << ")";
  }
  if (auto ret = cudaStreamDestroy(_cuda_stream)) {
    ERROR << "consumer:Failed to destroy cuda stream. (code: " << ret << ")";
  }
  _cuda_stream = nullptr;

  return true;
}

}  // namespace nvcvcam
