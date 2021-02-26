/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "frame.hpp"
#include "nvcvcam_error.hpp"
#include "utils.hpp"

#ifdef USE_NPP

#include <npp.h>
#include <nppi_color_conversion.h>
#include <limits>

#else  // USE_NPP

#include <opencv2/cudaimgproc.hpp>

#endif  // USE_NPP

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

  if (_raw_frame.width > std::numeric_limits<int>::max()) {
    ERROR << "_raw_frame.width out of range: " << _raw_frame.width;
    std::terminate();
  }

  if (_raw_frame.height > std::numeric_limits<int>::max()) {
    ERROR << "_raw_frame.height out of range: " << _raw_frame.height;
    std::terminate();
  }

  if (!utils::printCUDAEGLFrame(_raw_frame)) {
    return;
  }

  // map the data to a GpuMat
  _mat = cv::cuda::GpuMat(_raw_frame.height, _raw_frame.width, CV_16UC1,
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
                          int code,
                          cv::cuda::Stream& stream) {
  // reallocate out if necessary
  out.create(_raw_frame.width, _raw_frame.height, CV_16UC4);
  DEBUG << "frame:out.step == " << out.step;
  DEBUG << "frame:_raw_frame.pitch == " << _raw_frame.pitch;
#ifdef USE_NPP
  cudaStream_t nppstream = nullptr;
  NppStreamContext nppctx{};
  NppStatus status;
  NppiSize in_size{
      .width = static_cast<int>(_raw_frame.width),
      .height = static_cast<int>(_raw_frame.height),
  };
  NppiRect roi{
      .x = 0,
      .y = 0,
      .width = in_size.width,
      .height = in_size.height,
  };
  const int rgba_to_brga[4] = {2, 1, 0, 3};

  // check we're using the right stream
  nppstream = nppGetStream();
  if (nppstream != (cudaStream_t)stream.cudaPtr()) {
    DEBUG << "frame:Resetting NPP CUDA stream to: " << stream.cudaPtr();
    status = nppSetStream((cudaStream_t)stream.cudaPtr());
    if (NPP_NO_ERROR != status) {
      WARNING << "frame:Could not reset NPP CUDA stream because: "
              << error_string(status) << ". Performance may suffer.";
    }
  }

  // this will include the stream we just set(?) above.
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

  // debayer to 16 bit RGBA
  // FIXME(mdegans): map debayer order from OpenCV to NPP
  status = nppiCFAToRGBA_16u_C1AC4R_Ctx(
      (Npp16u*)_raw_frame.frame.pPitch[0], _raw_frame.pitch, in_size, roi,
      (Npp16u*)out.cudaPtr(), out.step, NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED,
      std::numeric_limits<ushort>::max(), nppctx);
  if (NPP_SUCCESS != status) {
    ERROR << "frame:Could not convert `out` to RGBA because: "
          << error_string(status) << ".";
    // FIXME(mdegans): throws NPP_STEP_ERROR, which says either step must be
    //  zero, but I can confirm it's not, and there's no source to examine so
    //  F this. I'll just use OpenCV since it doesn't make me want to jump off
    //  a building. LOLS. looking at the source, OpenCV uses npp internally.
    return false;
  }

  // swap B and R channels in place.
  status = nppiSwapChannels_16u_C4IR_Ctx((Npp16u*)out.cudaPtr(), out.step,
                                         in_size, rgba_to_brga, nppctx);
  if (NPP_SUCCESS != status) {
    ERROR << "frame:Could could not swap B and R channels becuase: "
          << error_string(status) << ".";
    return false;
  }
#else
  if (!sync()) {
    return false;
  }
  // just one line, this is GLORIOUS. Npp may be fast, but readability beats
  // speed and OpenCV can be optimized were Nvidia interested in doing so.
  cv::cuda::demosaicing(_mat, out, code, out.channels(), stream);
#endif
  return true;
}

cv::Size Frame::size() {
  return cv::Size(static_cast<int>(_raw_frame.width),
                  static_cast<int>(_raw_frame.height));
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
