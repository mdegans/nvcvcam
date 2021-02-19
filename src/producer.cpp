/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2021 Michael de Gans. All rights reserved.
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

#include "producer.hpp"

#include "stoppable_thread.hpp"

#include "nvcvcam_error.hpp"

#include "utils.hpp"

namespace nvcvcam {

bool Producer::setup() {
  Argus::Status err;  // argus error status (0 is success)

  // reset everything, just in case
  if (!cleanup()) {
    ERROR << "setup failed since cleanup could not be performed";
    return false;
  }

  // TODO(mdegans): more macros

  DEBUG << "getting camera provider";
  _provider.reset(Argus::CameraProvider::create(&err));
  if (err) {
    ERROR << "could not create camera provider (status" << err << ")";
    return false;
  }
  _iprovider = Argus::interface_cast<Argus::ICameraProvider>(_provider);

  DEBUG << "getting camera device " << _csi_id << "from provider";
  _device = utils::getCameraDevice(_iprovider, _csi_id);
  if (!_device) {
    ERROR << "setup failed since could not open csi id " << _csi_id;
    return false;
  }

  DEBUG << "creating capture session for csi camera " << _csi_id;
  _session.reset(_iprovider->createCaptureSession(_device, &err));
  if (err) {
    ERROR << "failed to create CaptureSession (status " << err << ")";
    return false;
  }
  _isession = Argus::interface_cast<Argus::ICaptureSession>(_session);
  if (!_isession) {
    ERROR << "could not get ICaptureSession interface";
    return false;
  }

  DEBUG << "Creating bayer output stream settings.";
  _settings.reset(
      _isession->createOutputStreamSettings(Argus::STREAM_TYPE_EGL, &err));
  if (err) {
    ERROR << "failed to create OutputStreamSettings (status " << err << ")";
    return false;
  }
  _isettings =
      Argus::interface_cast<Argus::IEGLOutputStreamSettings>(_settings);
  if (!_isettings) {
    ERROR << "failed to get IEGLOutputStreamSettings";
    return false;
  }

  err = _isettings->setPixelFormat(Argus::PIXEL_FMT_RAW16);
  if (err) {
    ERROR << "failed to set RAW16 pixel format";
    return false;
  }

  err = _isettings->setMode(Argus::EGL_STREAM_MODE_FIFO);
  if (err) {
    ERROR << "failed to set EGL_STREAM_MODE_FIFO";
    return false;
  }

  _stream.reset(_isession->createOutputStream(_settings.get(), &err));
  if (err) {
    ERROR << "failed to create OutputStream";
    return false;
  }

  _istream = Argus::interface_cast<Argus::IEGLOutputStream>(_stream);
  if (_istream) {
    ERROR << "could not get IEGLOutputStream from OutputStream";
    return false;
  }

  DEBUG << "Creating capture request.";
  // NOTE: manual disables autofocus and awb
  _request.reset(_isession->createRequest(Argus::CAPTURE_INTENT_MANUAL, &err));
  if (err) {
    ERROR << "unable to create capture request (status " << err << ")";
    return false;
  }
  _irequest = Argus::interface_cast<Argus::IRequest>(_request);
  if (!_irequest) {
    ERROR << "failed to get IRequest interface from Request";
    return false;
  }

  DEBUG << "enabling OutputStream for request";
  if (Argus::Status::STATUS_OK !=
      _irequest->enableOutputStream(_stream.get())) {
    DEBUG << "failed to enable OutputStream for request";
    return false;
  }
  _irequest = Argus::interface_cast<Argus::IRequest>(_request);
  if (!_irequest) {
    ERROR << "failed to get IRequest interface from Request";
    return false;
  }
  _isourcesettings = Argus::interface_cast<Argus::ISourceSettings>(_request);
  if (!_isourcesettings) {
    ERROR << "failed to get ISourceSettings interface from Request";
    return false;
  }

  // set the requested mode
  if (!set_mode(_csi_id)) {
    return false;
  }

  // success
  return true;
}

bool Producer::cleanup() {
  // cleanup any camera provider and interface
  if (_provider) {
    _provider.reset(nullptr);
  }
  _iprovider = nullptr;

  // reset device
  _device = nullptr;

  // reset any modes
  _mode = nullptr;
  _imode = nullptr;

  // cleanup any session
  if (_isession) {
    _isession->cancelRequests();
    _isession->waitForIdle();
  }
  _isession = nullptr;
  if (_session) {
    _session.reset(nullptr);
  }

  // cleanup capture seettings
  _isettings = nullptr;
  _settings.reset(nullptr);

  // cleanup any stream
  if (_istream) {
    _istream->disconnect();
  }
  _istream = nullptr;
  if (_stream) {
    _stream.reset(nullptr);
  }

  // cleanup any request
  _irequest = nullptr;
  _isourcesettings = nullptr;
  _request.reset(nullptr);
}

Argus::ICameraProperties* Producer::get_properties() {
  if (!_device) {
    ERROR << "no device to get properties from";
    return nullptr;
  }

  auto properties = Argus::interface_cast<Argus::ICameraProperties>(_device);
  if (!properties) {
    ERROR << "could not get ICameraProperties interface from _device";
    return nullptr;
  }
}

std::vector<Argus::SensorMode*> Producer::get_modes() {
  auto modes = std::vector<Argus::SensorMode*>();

  auto properties = get_properties();
  if (!properties) {
    return modes;
  }

  if (Argus::STATUS_OK != properties->getAllSensorModes(&modes)) {
    ERROR << "could not get sensor modes from csi id" << _csi_id;
    return modes;
  }

  if (modes.size() == 0) {
    ERROR << "no sensor modes are available for csi id" << _csi_id;
    return modes;
  }

  return modes;
}

bool Producer::set_mode(Argus::SensorMode* mode) {
  Argus::Status err;

  auto imode = Argus::interface_cast<Argus::ISensorMode>(mode);
  if (imode) {
    ERROR << "could not get ISensorMode interface from mode";
    return false;
  }

  auto res = _imode->getResolution();
  err = _isettings->setResolution(res);
  if (err) {
    ERROR << "streams settings would not accept resolution: " << res.width()
          << "x" << res.height() << "(status " << err << ")";
    return false;
  }

  DEBUG << "setting SensorMode on Request";
  err = _isourcesettings->setSensorMode(mode);
  if (err) {
    ERROR << "could not set SensorMode on Request (status " << err << ")";
    return false;
  }

  _mode = mode;
  _imode = imode;

  return true;
}

bool Producer::set_mode(uint32_t csi_mode) {
  auto modes = get_modes();
  if (modes.empty()) {
    return false;
  }

  if (modes.size() <= csi_mode) {
    ERROR << "requested csi_id does not exist. valid: 0 to "
          << modes.size() - 1;
    return false;
  }

  return set_mode(modes[csi_mode]);
}

// bool Producer::enqueue_request(uint64_t timeout_ns, repeatin) {
//   Argus::Status err;

//   if (!(ready())) {
//     ERROR << "producer not ready";
//     return false;
//   }

//   // request a capture
//   _isession->capture(_request.get(), timeout_ns, &err);
//   if (err) {
//     ERROR << "producer could not request a capture (status " << err << ")";
//     return false;
//   }

//   // success
//   return true;
// }

Argus::ISensorMode* Producer::get_imode() {
  if (!_imode) {
    ERROR << "cannot get mode since no mode is yet set";
    return nullptr;
  }
  return _imode;
}

bool Producer::get_resolution(Argus::Size2D<uint32_t>& out) {
  if (!_imode) {
    ERROR << "cannot get resolution since no mode is set yet";
    return false;
  }
  out = _imode->getResolution();
  return true;
}

Argus::OutputStream* Producer::get_output_stream() {
  if (!_stream) {
    ERROR << "cannot get OutputStream since not yet created";
    return nullptr;
  }
  return _stream.get();
}

}  // namespace nvcvcam