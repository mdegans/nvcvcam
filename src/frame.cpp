/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "frame.hpp"
#include "demosaic_kernel.hpp"
#include "nvcvcam_error.hpp"

namespace nvcvcam {

Frame::Frame(CUgraphicsResource resource,
             CUeglStreamConnection conn,
             cudaStream_t stream)
    : _resource(resource), _conn(conn), _stream(stream), _raw_frame{}, _mat() {
  CUresult cu_err;

  DEBUG << "frame:Mapping from resource.";
  cu_err = cuGraphicsResourceGetMappedEglFrame(&_raw_frame, resource, 0, 0);
  if (cu_err) {
    ERROR << "frame:Could not map CUgraphicsResource to CUeglFrame.";
    return;
  }

  // map the data to a GpuMat
  _mat = cv::cuda::GpuMat(_raw_frame.height, _raw_frame.width, CV_8UC4,
                          _raw_frame.frame.pPitch[0]);
}

bool Frame::get_debayered(cv::cuda::GpuMat& out,
                          const DebayerGains& gains,
                          cv::cuda::Stream& stream) {
  (void)gains;

  out.create(CV_8UC4, _raw_frame.width / 2, _raw_frame.height / 2);

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
    ERROR << "frame:Could release resource.";
    std::terminate();
  }
}

}  // namespace nvcvcam
