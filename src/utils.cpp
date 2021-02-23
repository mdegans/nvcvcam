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

#include "utils.hpp"
#include "nvcvcam_error.hpp"

/**
 * Modified copypasta from the MMAPI samples.
 */
namespace nvcvcam::utils {

bool printCUDAEGLFrame(const CUeglFrame& cudaEGLFrame) {
  DEBUG << "CUeglFrame:";
  DEBUG << " width: " << cudaEGLFrame.width;
  DEBUG << " height: " << cudaEGLFrame.height;
  DEBUG << " depth: " << cudaEGLFrame.depth;
  DEBUG << " pitch: " << cudaEGLFrame.pitch;
  DEBUG << " planeCount: " << cudaEGLFrame.planeCount;
  DEBUG << " numChannels: " << cudaEGLFrame.numChannels;
  const char* frameTypeString = NULL;
  switch (cudaEGLFrame.frameType) {
    case CU_EGL_FRAME_TYPE_ARRAY:
      frameTypeString = "array";
      break;
    case CU_EGL_FRAME_TYPE_PITCH:
      frameTypeString = "pitch";
      break;
    default:
      ERROR << "Unknown frame type " << cudaEGLFrame.frameType;
      return false;
  }
  DEBUG << " frameType: " << frameTypeString;
  const char* colorFormatString = NULL;
  switch (cudaEGLFrame.eglColorFormat) {
    case CU_EGL_COLOR_FORMAT_YUV420_PLANAR:
      colorFormatString = "YUV420 planar";
      break;
    case CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR:
      colorFormatString = "YUV420 semi-planar";
      break;
    case CU_EGL_COLOR_FORMAT_YUV422_PLANAR:
      colorFormatString = "YUV422 planar";
      break;
    case CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR:
      colorFormatString = "YUV422 semi-planar";
      break;
    case CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER:
      colorFormatString = "YUV420 planar ER";
      break;
    case CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER:
      colorFormatString = "YUV420 semi-planar ER";
      break;
    case CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER:
      colorFormatString = "YUV422 planar ER";
      break;
    case CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER:
      colorFormatString = "YUV422 semi-planar ER";
      break;
    case CU_EGL_COLOR_FORMAT_RGB:
      colorFormatString = "RGB";
      break;
    case CU_EGL_COLOR_FORMAT_BGR:
      colorFormatString = "BGR";
      break;
    case CU_EGL_COLOR_FORMAT_ARGB:
      colorFormatString = "ARGB";
      break;
    case CU_EGL_COLOR_FORMAT_RGBA:
      colorFormatString = "RGBA";
    case CU_EGL_COLOR_FORMAT_BAYER_RGGB:
      colorFormatString = "S16 Bayer RGGB";
      break;
    case CU_EGL_COLOR_FORMAT_BAYER_BGGR:
      colorFormatString = "S16 Bayer BGGR";
      break;
    case CU_EGL_COLOR_FORMAT_BAYER_GRBG:
      colorFormatString = "S16 Bayer GRBG";
      break;
    case CU_EGL_COLOR_FORMAT_BAYER_GBRG:
      colorFormatString = "S16 Bayer GBRG";
      break;
    default:
      ERROR << "Unknown color format " << cudaEGLFrame.eglColorFormat;
      return false;
  }
  DEBUG << " colorFormat: " << colorFormatString;
  const char* cuFormatString = NULL;
  switch (cudaEGLFrame.cuFormat) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
      cuFormatString = "uint8";
      break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
      cuFormatString = "uint16";
      break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
      cuFormatString = "uint32";
      break;
    case CU_AD_FORMAT_SIGNED_INT8:
      cuFormatString = "int8";
      break;
    case CU_AD_FORMAT_SIGNED_INT16:
      cuFormatString = "int16";
      break;
    case CU_AD_FORMAT_SIGNED_INT32:
      cuFormatString = "int32";
      break;
    case CU_AD_FORMAT_HALF:
      cuFormatString = "float16";
      break;
    case CU_AD_FORMAT_FLOAT:
      cuFormatString = "float32";
      break;
    default:
      ERROR << "Unknown cuFormat " << cudaEGLFrame.cuFormat;
  }
  DEBUG << " cuFormat: " << cuFormatString;

  return true;
}

bool init_cuda(CUcontext* ctx) {
  CUresult err;

  if (!ctx) {
    ERROR << "init_cuda:CUDA context NULL.";
    return false;
  }

  err = cuInit(0);
  if (err) {
    ERROR << "init_cuda:Unable to initialize the CUDA driver API because: "
          << error_string(err) << ".";
    return false;
  }

  int dev_count = 0;
  err = cuDeviceGetCount(&dev_count);
  if (err) {
    ERROR << "init_cuda:Unable to get CUDA device count because: "
          << error_string(err) << ".";
    return false;
  }

  if (!dev_count) {
    ERROR << "init_cuda:Unable to find any CUDA devices.";
    return false;
  }

  CUdevice dev;
  err = cuDeviceGet(&dev, 0);
  if (err) {
    ERROR << "init_cuda:Unable to get CUDA device 0 because: "
          << error_string(err) << ".";
    return false;
  }

  err = cuCtxCreate(ctx, 0, dev);
  if (err) {
    ERROR << "init_cuda:Unable to create CUDA context because: "
          << error_string(err) << ".";
    return false;
  }

  return true;
}

Argus::CameraDevice* getCameraDevice(Argus::ICameraProvider* iProvider,
                                     uint32_t csi_id) {
  Argus::Status status;
  std::vector<Argus::CameraDevice*> devices;

  if (!iProvider) {
    ERROR << "invalid argument. no ICameraProvider supplied";
    return nullptr;
  }

  status = iProvider->getCameraDevices(&devices);
  if (status != Argus::STATUS_OK) {
    ERROR << "failed to get camera devices from provider because: "
          << "status " << status;
    return nullptr;
  }
  if (devices.size() == 0) {
    ERROR << "no camera devices are available";
    return nullptr;
  }
  if (devices.size() <= csi_id) {
    ERROR << "requested csi_id does not exist. valid: 0 to "
          << devices.size() - 1;
    return nullptr;
  }

  return devices[csi_id];
}

Argus::SensorMode* getSensorMode(Argus::CameraDevice* cameraDevice,
                                 uint32_t csi_mode) {
  std::vector<Argus::SensorMode*> modes;
  Argus::ICameraProperties* properties;
  Argus::Status status;

  properties = Argus::interface_cast<Argus::ICameraProperties>(cameraDevice);
  if (!properties) {
    ERROR << "Failed to get ICameraProperties interface";
    return nullptr;
  }

  status = properties->getAllSensorModes(&modes);
  if (status != Argus::STATUS_OK) {
    ERROR << "Failed to get sensor modes from device.";
    return nullptr;
  }

  if (modes.size() == 0) {
    ERROR << "No sensor modes are available.";
    return nullptr;
  }

  if (modes.size() <= csi_mode) {
    ERROR << "requested csi_id does not exist. valid: 0 to "
          << modes.size() - 1;
    return nullptr;
  }

  return modes[csi_mode];
}
}  // namespace nvcvcam::utils
