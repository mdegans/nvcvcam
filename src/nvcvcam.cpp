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

using namespace Argus;

/**
 * Modified copypasta from the MMAPI samples.
 */
namespace ArgusHelpers {

/**
 * @brief Get the Camera Device object from a CameraProvider.
 *
 * @param cameraProvider
 * @param cameraDeviceIndex
 *
 * @return Argus::CameraDevice*
 */
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
          << error_string(status);
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

/**
 * @brief Get the SensorMode from a CameraDevice.
 *
 * @param cameraDevice device to get the sensor mode from
 * @param csi_mode requested sensor mode
 *
 * @return Argus::SensorMode*
 */
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

}  // namespace ArgusHelpers

namespace nvcvcam {

void NvCvCam::producer() {
  INFO << "producer starting...";

  this->stopping = false;
  std::mutex cv_lock;
  ICameraProvider* iProvider = nullptr;
  CameraDevice* device = nullptr;
  SensorMode* mode = nullptr;
  ISensorMode* iMode = nullptr;
  ICaptureSession* iSession = nullptr;
  IRequest* iRequest = nullptr;
  ISourceSettings* iSourceSettings = nullptr;

  IEGLOutputStreamSettings* iSettings = nullptr;
  IEGLOutputStream* iStream = nullptr;
  this->producer_status = Argus::Status::STATUS_DISCONNECTED;

  // reset our stream if any
  if (this->raw_stream) {
    WARNING << "OuputStream may not have been cleanly closed last time";
    this->raw_stream.reset();
  }

  DEBUG << "creating CameraProvider";
  Argus::Status ret;
  UniqueObj<CameraProvider> provider(CameraProvider::create(&ret));
  this->producer_status = ret;
  if (!provider) {
    ERROR << "could not create camera provider (no csi devices?)";
    return;
  }

  DEBUG << "getting ICameraProvider interface";
  iProvider = Argus::interface_cast<Argus::ICameraProvider>(provider);
  if (!iProvider) {
    ERROR << "failed to get ICameraProvider interface";
    return;
  }

  DEBUG << "getting CameraDevice from CameraProvider";
  device = ArgusHelpers::getCameraDevice(iProvider, this->csi_id);
  if (!device) {
    ERROR << "csi device " << this->csi_id << " is currently unavailable.";
    this->producer_status = Argus::Status::STATUS_UNAVAILABLE;
    return;
  }

  DEBUG << "getting SensorMode (" << this->csi_mode
        << ") from CameraDevice mode: ";
  mode = ArgusHelpers::getSensorMode(device, this->csi_mode);
  iMode = interface_cast<ISensorMode>(mode);
  if (!iMode) {
    ERROR << "selected sensor mode (" << this->csi_mode << ") is unavailable";
    this->producer_status = Argus::Status::STATUS_INVALID_SETTINGS;
    return;
  }

  DEBUG << "creating capture session for csi camera " << this->csi_id;
  UniqueObj<CaptureSession> session(
      iProvider->createCaptureSession(device, &ret));
  this->producer_status = ret;
  iSession = interface_cast<ICaptureSession>(session);
  if (!iSession) {
    ERROR << "failed to create CaptureSession (status " << ret << ")";
    return;
  }

  DEBUG << "Creating bayer output stream settings.";
  UniqueObj<OutputStreamSettings> settings(
      iSession->createOutputStreamSettings(STREAM_TYPE_EGL, &ret));
  this->producer_status = ret;
  iSettings = interface_cast<IEGLOutputStreamSettings>(settings);
  if (!iSettings) {
    ERROR << "Failed to create OutputStreamSettings (status " << ret << ")";
    return;
  }

  // set various output stream settings

  this->producer_status = iSettings->setPixelFormat(PIXEL_FMT_RAW16);
  if (this->producer_status == Argus::Status::STATUS_OK) {
    DEBUG << "set pixel format to RAW16.";
  } else {
    WARNING << "failed to set pixel format to RAW16";
  }

  auto resolution = iMode->getResolution();
  this->producer_status = iSettings->setResolution(resolution);
  if (this->producer_status == Argus::Status::STATUS_OK) {
    DEBUG << "set resolution to " << resolution.width() << "x"
          << resolution.height();
  } else {
    WARNING << "failed to set resolution to " << resolution.width() << "x"
            << resolution.height();
  }

  this->producer_status = iSettings->setMode(EGL_STREAM_MODE_FIFO);
  if (this->producer_status == Argus::Status::STATUS_OK) {
    DEBUG << "egl stream mode set to EGL_STREAM_MODE_FIFO";
  } else {
    WARNING << "failed to set mode set to EGL_STREAM_MODE_FIFO";
  }

  DEBUG << "Creating bayer OutputStream.";
  this->raw_stream = UniqueObj<OutputStream>(
      iSession->createOutputStream(settings.get(), &ret));
  this->producer_status = ret;
  iStream = interface_cast<IEGLOutputStream>(raw_stream);
  if (!iStream) {
    ERROR << "failed to create OutputStream";
    return;
  }

  DEBUG << "Creating capture request.";
  UniqueObj<Request> request(
      iSession->createRequest(CAPTURE_INTENT_MANUAL, &ret));
  this->producer_status = ret;
  iRequest = interface_cast<IRequest>(request);
  if (!iRequest) {
    ERROR << "unable to create capture request (status " << ret << ")";
    return;
  }

  DEBUG << "enabling OutputStream";
  iRequest->enableOutputStream(raw_stream.get());
  DEBUG << "OutputStream enabled";

  DEBUG << "getting ISourceSettings interface from Request";
  iSourceSettings = interface_cast<ISourceSettings>(request);
  if (!iSourceSettings) {
    ERROR << "failed to get ISourceSettings interface from Request";
    return;
  }
  DEBUG << "setting SensorMode on Request";
  this->producer_status = iSourceSettings->setSensorMode(mode);
  if (this->producer_status != Argus::Status::STATUS_OK) {
    WARNING << "could not set SensorMode (" << this->csi_mode << ") on Request";
  }

  // start repeated requests
  DEBUG << "making repeated capture requests.";
  this->producer_status = iSession->repeat(request.get());
  if (this->producer_status != Argus::Status::STATUS_OK) {
    ERROR << "failed to make repeat requests (status " << this->producer_status
          << ")";
  }

  std::unique_lock<std::mutex> lock(cv_lock);
  INFO << "producer OutputStream ready";
  this->stream_ready.notify_all();
  DEBUG << "producer waiting for shutdown.";
  this->shutdown_requested.wait(lock);

  // cleanup
  INFO << "producer cleaning up...";
  this->producer_status = iSession->cancelRequests();
  DEBUG << "CaptureSession capture requests cancelled.";
  this->raw_stream.reset();
  this->producer_status = Argus::Status::STATUS_DISCONNECTED;
  DEBUG << "producer shutdown complete.";
  this->stopping = false;
  return;
}

// class ConsumerCtx {
//   std::mutex cv_lock;
//   Argus::Status ret;
//   IEGLOutputStream* i_raw_stream;
//   CUeglStreamConnection cuda_connection;
//   cudaStream_t cuda_stream;
//   Argus::Size2D<uint32_t> out_resolution;

//   ConsumerCtx(): i_raw_stream(nullptr),
// }

void NvCvCam::consumer() {
  INFO << "consumer starting...";

  std::mutex cv_lock;
  CUresult cu_ret;
  IEGLOutputStream* i_raw_stream = nullptr;
  CUeglStreamConnection cuda_connection;
  cudaStream_t cuda_stream;

  DEBUG << "consumer creating cuda stream";
  if (!CUDA_OK(cudaStreamCreate(&cuda_stream))) {
    ERROR << "consumer failed to create cuda stream";
    return;
  }

  DEBUG << "consumer waiting for producer";
  std::unique_lock<std::mutex> lock(cv_lock);
  this->stream_ready.wait(lock);

  DEBUG << "checking for stream";
  if (!this->raw_stream) {
    ERROR << "consumer did not find an egl stream";
    // FIXME(mdegans): condition variables are not events so this might be racey
    // and the producer might never get this notification. I did this becuase
    // I'm not sure what'll hapen if I do "while (!stopping) request frame"
    // will it block, or will Argus crap out because I request 10000000 frames?
    // or should I make requests from the consumer thread?
    this->shutdown_requested.notify_all();
    cudaStreamDestroy(cuda_stream);
    return;
  }

  DEBUG << "getting IEGLOutputStream interface from OutputStream";
  i_raw_stream = interface_cast<IEGLOutputStream>(this->raw_stream);
  if (!i_raw_stream) {
    ERROR << "could not get IEGLOutputStream interface";
    this->shutdown_requested.notify_all();
    cudaStreamDestroy(cuda_stream);
    return;
  }

  INFO << "consumer connecting to OutputStream";
  this->consumer_status = i_raw_stream->waitUntilConnected();
  if (this->consumer_status != Argus::Status::STATUS_OK) {
    ERROR << "consumer could not connect to OutputStream (status "
          << this->consumer_status << ")";
    this->shutdown_requested.notify_all();
    cudaStreamDestroy(cuda_stream);
    return;
  }

  cu_ret = cuEGLStreamConsumerConnect(&cuda_connection,
                                      i_raw_stream->getEGLStream());
  if (cu_ret != CUDA_SUCCESS) {
    ERROR << "consumer unable to connect CUDA to OutputStream because: "
          << error_string(cu_ret);
    this->shutdown_requested.notify_all();
    cudaStreamDestroy(cuda_stream);
    return;
  }

  INFO << "consumer connected";
  while (!this->stopping) {
    CUgraphicsResource bayer_resource = 0;
    CUeglFrame bayer_frame;

    DEBUG << "getting an image from the EGLStream";
    cu_ret = cuEGLStreamConsumerAcquireFrame(&cuda_connection, &bayer_resource,
                                             &cuda_stream, -1);
    if (cu_ret != CUDA_SUCCESS) {
      ERROR << "consumer failed to get image from the EglStream becuase: "
            << error_string(cu_ret);
      break;
    }

    cu_ret =
        cuGraphicsResourceGetMappedEglFrame(&bayer_frame, bayer_resource, 0, 0);
    if (cu_ret != CUDA_SUCCESS) {
      ERROR << "could not map image to CUeglFrame because: "
            << error_string(cu_ret);
      break;
    }

    // Sanity check for one of the required input color formats.
    if ((bayer_frame.cuFormat != CU_AD_FORMAT_SIGNED_INT16) ||
        ((bayer_frame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_RGGB) &&
         (bayer_frame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_BGGR) &&
         (bayer_frame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_GRBG) &&
         (bayer_frame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_GBRG))) {
      ERROR << "Only 16bit signed Bayer color formats are supported";
      break;
    }

    // create our bgra frame
    cv::cuda::GpuMat out(bayer_frame.height / 2, bayer_frame.width / 2,
                         CV_8UC4);

    // convert the bayer frame to rgba
    cudaBayerDemosaic((CUdeviceptr)bayer_frame.frame.pPitch[0],
                      bayer_frame.width, bayer_frame.height, bayer_frame.pitch,
                      bayer_frame.eglColorFormat, cuda_stream,
                      (CUdeviceptr)out.cudaPtr());

    // Return the Bayer frame to the Argus stream.
    cu_ret = cuEGLStreamConsumerReleaseFrame(&cuda_connection, bayer_resource,
                                             &cuda_stream);
    if (cu_ret != CUDA_SUCCESS) {
      ERROR << "could not release egl frame becuase: " << error_string(cu_ret);
      break;
    }

    // synchronize stream
    if (!CUDA_OK(cudaStreamSynchronize(cuda_stream))) {
      ERROR << "consumer could not synchronize cuda stream";
    }

    // store the latest mat
    std::unique_lock<std::mutex> lock(this->mat_lock);
    this->latest_mat = out;
    this->capture_ready.notify_all();
  }

  DEBUG << "consumer notifying producer that we're done";
  this->shutdown_requested.notify_all();
  if (!CUDA_OK(cudaStreamDestroy(cuda_stream))) {
    ERROR << "could not destory consumer's cuda stream";
  }
  DEBUG << "consumer shutdown complete.";
  return;
}

bool NvCvCam::open(uint32_t csi_id, uint32_t csi_mode) {
  if (this->producer_thread || this->consumer_thread) {
    if (!this->close()) {
      ERROR << "could not open because threads are stuck and won't close";
      return false;
    }
  }

  // set csi id and modes
  this->csi_id = csi_id;
  this->csi_mode = csi_mode;

  // reset stop triger
  this->stopping = false;

  // spawn worker threads
  this->producer_thread =
      std::make_unique<std::thread>(&NvCvCam::producer, this);
  this->consumer_thread =
      std::make_unique<std::thread>(&NvCvCam::consumer, this);

  // wait until capture is ready
  std::mutex cv_lock;
  std::unique_lock<std::mutex> lock(cv_lock);
  this->capture_ready.wait(lock);

  return true;
}

bool NvCvCam::close() {
  this->stopping = true;
  if (this->producer_thread->joinable()) {
    this->producer_thread->join();
  }
  if (this->consumer_thread->joinable()) {
    this->consumer_thread->join();
  }
  this->stopping = false;
  return true;
}

bool NvCvCam::read(cv::cuda::GpuMat& out) {
  std::unique_lock<std::mutex> lock(this->mat_lock);
  if (this->latest_mat.empty()) {
    ERROR << "latest_mat empty";
    return false;
  }

  out = latest_mat;
  return true;
}

}  // namespace nvcvcam
