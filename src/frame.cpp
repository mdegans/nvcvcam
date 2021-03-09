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

#include <map>

namespace nvcvcam {

// static const std::map<Format, int> FORMAT_TO_CV({
//     {Format::BAYER, CV_16UC1},
//     {Format::Y8, CV_8UC1},
//     {Format::Y16, CV_16UC1},
//     {Format::YUV420, CV_8UC1},
//     {Format::NV12, CV_8UC1},
//     {Format::P016, CV_16UC1},
// });

static const std::map<CUarray_format, int> CU_FORMAT_TO_CV({
    {CU_AD_FORMAT_UNSIGNED_INT8, CV_8U},
    {CU_AD_FORMAT_UNSIGNED_INT16, CV_16U},
    // 16I is mapped to 16U because this appears to be the actual format.
    // 16I gives incorrect results, even when converted to 16U using
    // mat_a.convertTo(mat_b, CV_16UC1, 2.0);
    {CU_AD_FORMAT_SIGNED_INT16, CV_16U},
    {CU_AD_FORMAT_HALF, CV_16F},
    {CU_AD_FORMAT_FLOAT, CV_32F},
});

static inline int cv_type_of(const CUeglFrame& frame) {
  return CV_MAKE_TYPE(CU_FORMAT_TO_CV.at(frame.cuFormat), frame.numChannels);
}

Frame::Frame(CUgraphicsResource resource,
             CUeglStreamConnection conn,
             cudaStream_t stream,
             Format format)
    : format(format), _resource(resource), _conn(conn), _stream(stream) {
  CUresult cu_err;
  uint height;
  size_t step;

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

  if (format == Format::YUV420 || format == Format::NV12 ||
      format == Format::P016) {
    height = _raw_frame.height * 3U / 2U;
  } else {
    height = _raw_frame.height;
  }

  if (height > std::numeric_limits<int>::max()) {
    ERROR << "_raw_frame.height out of range: " << height;
    std::terminate();
  }

  if (!utils::printCUDAEGLFrame(_raw_frame)) {
    return;
  }

  if (_raw_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
    // Doesn't seem to make a difference. Still illegal access whenever
    // not using bayer.
    step = cv::Mat::CONTINUOUS_FLAG;
  } else if (_raw_frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
    step = static_cast<size_t>(_raw_frame.pitch);
  } else {
    // should never happen
    ERROR << "Invalid CUeglFrameType. Fatal.";
    std::terminate();
  }

  // map the data to a GpuMat
  _mat = cv::cuda::GpuMat(height, _raw_frame.width, cv_type_of(_raw_frame),
                          _raw_frame.frame.pArray[0], step);
}

bool Frame::sync() {
  DEBUG << "frame:Syncronizing cuda stream";
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
