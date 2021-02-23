/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * mdegans wuz here - adapted from cudaBayerDemosaic mmapi sample
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

#include "nvcvcam.hpp"
#include "demosaic_kernel.hpp"
#include "nvcvcam_error.hpp"
#include "utils.hpp"

#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <nvbuf_utils.h>

namespace nvcvcam {

bool NvCvCam::open(uint32_t csi_id, uint32_t csi_mode) {
  CUresult cu_err;
  Argus::Status status;

  INFO << "nvcvcam:Opening.";

  DEBUG << "nvcvcam:Initializing CUDA.";
  if (!utils::init_cuda(&_ctx)) {
    ERROR << "nvcvcam:Cuda initialization failed.";
    return false;
  }

  DEBUG << "nvcvcam:Creating CUDA stream.";
  auto errt = cudaStreamCreate(&_cuda_stream);
  if (errt) {
    ERROR << "nvcvcam:Could not create CUDA stream becuase: "
          << error_string(errt) << ".";
    return false;
  }

  // NOTE(medgans) order is important here. See `NOTE` in `close`.
  _producer.reset(new Producer(csi_id, csi_mode));
  if (!_producer->start()) {
    ERROR << "nvcvcam:Could not start Producer.";
    return false;
  }

  auto raw_stream = _producer->get_output_stream();
  auto iraw_stream = Argus::interface_cast<Argus::IEGLOutputStream>(raw_stream);
  if (!iraw_stream) {
    ERROR << "nvcvcam:Could not get OutputStream from Producer.";
  }

  DEBUG << "nvcvcam:Connecting to producer.";
  cu_err = cuEGLStreamConsumerConnect(&_cuda_conn, iraw_stream->getEGLStream());
  if (cu_err) {
    ERROR << "nvcvcam:Could not connect CUDA to OutputStream because: "
          << error_string(cu_err) << ".";
    return false;
  }
  status = iraw_stream->waitUntilConnected();
  if (status) {
    ERROR << "nvcvcam:Could not connect OutputStream becuase: (status "
          << status << ").";
  }
  DEBUG << "nvcvcam:Connected to producer.";

  return true;
}

bool NvCvCam::close() {
  CUresult err;
  bool success = true;

  if (!_producer) {
    ERROR << "nvcvcam:Camera is not yet open.";
    return false;
  }
  if (!_producer->ready()) {
    ERROR << "nvcvcam:Camera is not yet ready.";
    return false;
  }

  INFO << "nvcvcam:Closing camera.";
  success = _producer->stop();

  DEBUG << "nvcvcam:Disconnecting from producer.";
  err = cuEGLStreamConsumerDisconnect(&_cuda_conn);
  if (err) {
    ERROR << "nvcvcam:Could not disconnect from producer stream because: "
          << error_string(err) << ".";
    success = false;
  }

  DEBUG << "nvcvcam:Destroying CUDA stream.";
  auto errt = cudaStreamDestroy(_cuda_stream);
  if (errt) {
    ERROR << "nvcvcam:Could not destroy CUDA stream because: "
          << error_string(errt) << ".";
    success = false;
  }

  DEBUG << "nvcvcam:Destroying CUDA context.";
  err = cuCtxDestroy(_ctx);
  if (err) {
    ERROR << "nvcvcam:Could not destroy cuda context because: "
          << error_string(err) << ".";
    success = false;
  }

  return success;
}

std::unique_ptr<Frame> NvCvCam::capture() {
  CUgraphicsResource tmp_res = nullptr;

  if (!_producer) {
    ERROR << "nvcvcam:Camera is not yet open.";
    return nullptr;
  }
  if (!_producer->ready()) {
    ERROR << "nvcvcam:Camera is not yet ready.";
    return nullptr;
  }

  DEBUG << "nvcvcam:Getting an image from the EGLStream.";
  if (auto cu_err = cuEGLStreamConsumerAcquireFrame(&_cuda_conn, &tmp_res,
                                                    &_cuda_stream, -1)) {
    ERROR << "nvcvcam:Could not get image from the EglStream becuase: "
          << error_string(cu_err) << ".";
    return nullptr;
  }

  DEBUG << "nvcvcam:Returning Frame.";
  return std::make_unique<Frame>(tmp_res, _cuda_conn, _cuda_stream);
}

}  // namespace nvcvcam
