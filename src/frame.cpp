/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "frame.hpp"
#include "nvcvcam_error.hpp"
#include "utils.hpp"

#include <npp.h>
#include <nppi_color_conversion.h>

namespace nvcvcam {

// this is NVIDIA's. here temporarily

Frame::Frame(CUgraphicsResource resource,
             CUeglStreamConnection conn,
             cudaStream_t stream)
    : _resource(resource), _conn(conn), _stream(stream), _raw_frame(), _mat() {
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

  // map the data to a GpuMat
  _mat = cv::cuda::GpuMat(_raw_frame.height, _raw_frame.width, CV_16SC1,
                          _raw_frame.frame.pPitch[0], _raw_frame.pitch);
}

bool Frame::sync() {
  if (!_synced) {
    auto err = cudaStreamSynchronize(_stream);
    if (err) {
      ERROR << "frame:Could not synchronize cuda stream becuase: "
            << error_string(err) << ".";
      return false;
    } else {
      _synced = true;
    }
  }

  return true;
}

cv::cuda::GpuMat Frame::gpu_mat() {
  if (!sync()) {
    ERROR << "frame:Sync error. Returning empty GpuMat.";
    return cv::cuda::GpuMat();
  }
  return _mat;
}

bool Frame::get_debayered(cv::cuda::GpuMat& out,
                          const DebayerGains& gains,
                          cv::cuda::Stream& stream,
                          bool u16bpp = true) {
  (void)gains;
  (void)stream;
  cudaStream_t nppstream = nullptr;
  NppStreamContext nppctx();
  NppStatus status;
  NppiSize in_size{
      height = _raw_frame.height,
      width = _raw_frame.width,
  };
  NppiRect roi();

  out.create(_raw_frame.width, _raw_frame.height, CV_16UC4);

  // check we're using the right stream
  nppstream = nppGetStream();
  if (nppstream != (cudaStream_t)stream.cudaPtr()) {
    DEBUG << "frame:Resetting NPP CUDA stream.";
    status = nppSetStream((cudaStream_t)stream.cudaPtr());
    if (NPP_NO_ERROR != status) {
      WARNING << "frame:Could not reset NPP CUDA stream because: "
              << error_string(status) << ". Performance may suffer.";
    }
  }

  // get npp context
  status = nppGetStreamContext(&nppctx);
  if (NPP_SUCCESS != status) {
    ERROR << "frame:Could not get NPP stream context because: "
          << error_string(status) << ".";
    return false;
  }

  if (!sync()) {
    return false;
  }

  // convert the bayer frame to bgra
  // cudaBayerDemosaic((CUdeviceptr)_raw_frame.frame.pPitch[0],
  // _raw_frame.width,
  //                   _raw_frame.height, _raw_frame.pitch,
  //                   _raw_frame.eglColorFormat,
  //                   (cudaStream_t)stream.cudaPtr(),
  //                   (CUdeviceptr)out.cudaPtr());

  nppiCFAToRGB_16u_C1C3R((npp16u*)_raw_frame.frame.pPitch[0], _raw_frame.pitch,
                         in_size, roi, (npp16u*)out.cudaPtr().out.step1());

  return true;
}

Frame::~Frame() {
  CUresult cu_err;

  DEBUG << "frame:Releasing resource in stream: " << (size_t)_stream;
  DEBUG << "frame:Connection: " << _conn;
  DEBUG << "frame:Resource: " << _resource;
  cu_err = cuEGLStreamConsumerReleaseFrame(&_conn, _resource, &_stream);
  if (cu_err) {
    ERROR << "frame:Could not release resource because: "
          << error_string(cu_err) << "(" << cu_err << ").";
    std::terminate();
  }
}

}  // namespace nvcvcam
