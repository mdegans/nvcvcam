/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "frame.hpp"
#include "demosaic_kernel.hpp"
#include "nvcvcam_error.hpp"
#include "utils.hpp"

namespace nvcvcam {

// this is NVIDIA's. here temporarily

Frame::Frame(CUgraphicsResource resource,
             CUeglStreamConnection conn,
             cudaStream_t stream)
    : _resource(resource), _conn(conn), _stream(stream), _raw_frame{}, _mat() {
  CUresult cu_err;

  DEBUG << "frame:Mapping from resource.";
  cu_err = cuGraphicsResourceGetMappedEglFrame(&_raw_frame, resource, 0, 0);
  if (cu_err) {
    ERROR << "frame:Could not map CUgraphicsResource to CUeglFrame becuase: "
          << error_string(cu_err) << ".";
    return;
  }

  if (!utils::printCUDAEGLFrame(_raw_frame)) {
    return;
  }

  if (!sync()) {
    return;
  }

  // map the data to a GpuMat
  _mat = cv::cuda::GpuMat(_raw_frame.height, _raw_frame.width, CV_16SC1,
                          _raw_frame.frame.pPitch[0], _raw_frame.pitch);
}

bool Frame::sync() {
  auto err = cudaStreamSynchronize(_stream);
  if (err) {
    ERROR << "Could not synchronize cuda stream becuase: " << error_string(err)
          << ".";
    return false;
  }
  return true;
}

bool Frame::get_debayered(cv::cuda::GpuMat& out,
                          const DebayerGains& gains,
                          cv::cuda::Stream& stream) {
  (void)gains;

  out.create(CV_8UC4, _raw_frame.width / 2, _raw_frame.height / 2);

  if (!sync()) {
    return false;
  }

  // convert the bayer frame to bgra
  cudaBayerDemosaic((CUdeviceptr)_raw_frame.frame.pPitch[0], _raw_frame.width,
                    _raw_frame.height, _raw_frame.pitch,
                    _raw_frame.eglColorFormat, (cudaStream_t)stream.cudaPtr(),
                    (CUdeviceptr)out.cudaPtr());

  return true;
}

Frame::~Frame() {
  CUresult cu_err;

  DEBUG << "frame:releasing resource in stream " << (size_t)_stream;
  cu_err = cuEGLStreamConsumerReleaseFrame(&_conn, _resource, &_stream);
  if (cu_err) {
    ERROR << "frame:Could not release resource because: "
          << error_string(cu_err) << "(" << cu_err << ").";
    std::terminate();
  }
}

}  // namespace nvcvcam
