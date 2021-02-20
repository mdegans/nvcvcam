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

#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <nvbuf_utils.h>

namespace nvcvcam {

bool NvCvCam::open(uint32_t csi_id,
                   uint32_t csi_mode,
                   bool block,
                   std::chrono::nanoseconds timeout) {
  _producer.reset(new Producer(csi_id, csi_mode));
  if (!_producer->start(block, timeout)) {
    ERROR << "nvcvcam:Could not start Producer.";
    return false;
  }
  _consumer.reset(new Consumer(_producer->get_output_stream()));
  if (!_consumer->start(block, timeout)) {
    ERROR << "nvcvcam:Could not start Consumer.";
    return false;
  }

  return true;
}

bool NvCvCam::close(bool block, std::chrono::nanoseconds timeout) {
  INFO << "nvcvcam:Closing camera.";
  return (_producer->stop(block, timeout) && _consumer->stop(block, timeout));
}

std::unique_ptr<Frame> NvCvCam::capture() {
  if (!(_producer && _consumer)) {
    ERROR << "nvcvcam:Camera is not yet open.";
    return nullptr;
  }
  if (!_producer->ready() && _consumer->ready()) {
    ERROR << "nvcvcam:Camera is not yet ready.";
    return nullptr;
  }
  DEBUG << "nvcvcam:Getting frame from consumer.";
  return _consumer->get_frame();
}

}  // namespace nvcvcam
