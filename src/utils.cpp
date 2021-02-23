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
