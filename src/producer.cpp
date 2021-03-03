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

#include <Argus/Ext/BayerSharpnessMap.h>

namespace nvcvcam {

bool Producer::setup() {
  Argus::Status err;  // argus error status (0 is success)

  DEBUG << "producer:Starting up.";

  // reset everything, just in case
  if (!cleanup()) {
    ERROR << "producer:Setup failed since cleanup could not be performed.";
    return false;
  }

  // TODO(mdegans): more macros

  DEBUG << "producer:Getting camera provider.";
  _provider.reset(Argus::CameraProvider::create(&err));
  if (err) {
    ERROR << "producer:Could not create camera provider (status" << err << ")";
    return false;
  }
  _iprovider = Argus::interface_cast<Argus::ICameraProvider>(_provider);

  DEBUG << "producer:Argus version: " << _iprovider->getVersion();

  DEBUG << "producer:Getting camera device " << _csi_id << " from provider.";
  _device = utils::getCameraDevice(_iprovider, _csi_id);
  if (!_device) {
    ERROR << "producer:Setup failed since could not open csi id " << _csi_id
          << ".";
    return false;
  }

  // FIXME(mdegans): this is copied frpm set_mode. remove this duplication.
  auto modes = get_modes();
  if (modes.size() <= _csi_mode) {
    ERROR << "producer:Requested csi_mode does not exist. Valid modes: 0 to "
          << modes.size() - 1 << ".";
    return false;
  }
  _mode = modes[_csi_mode];
  _imode = Argus::interface_cast<Argus::ISensorMode>(_mode);
  if (!_imode) {
    ERROR << "producer:Could not get ISensorMode interface from mode.";
    return false;
  }

  DEBUG << "producer:Creating CaptureSession for csi camera " << _csi_id << ".";
  _session.reset(_iprovider->createCaptureSession(_device, &err));
  if (err) {
    ERROR << "producer:Could not create CaptureSession (status " << err << ").";
    return false;
  }
  _isession = Argus::interface_cast<Argus::ICaptureSession>(_session);
  if (!_isession) {
    ERROR << "producer:Could not get ICaptureSession interface.";
    return false;
  }

  DEBUG << "producer:Creating bayer OutputStreamSettings.";
  _settings.reset(
      _isession->createOutputStreamSettings(Argus::STREAM_TYPE_EGL, &err));
  if (err) {
    ERROR << "producer:Could not create OutputStreamSettings (status " << err
          << ").";
    return false;
  }

  _isettings =
      Argus::interface_cast<Argus::IEGLOutputStreamSettings>(_settings);
  if (!_isettings) {
    ERROR << "producer:Could not get IEGLOutputStreamSettings.";
    return false;
  }

  // TODO(mdegans): add display here

  err = _isettings->setMetadataEnable(true);
  if (err) {
    ERROR << "producer:Could not enable capture metadata.";
    return false;
  }

  err = _isettings->setPixelFormat(Argus::PIXEL_FMT_RAW16);
  if (err) {
    ERROR << "producer:Could not set RAW16 pixel format.";
    return false;
  }

  err = _isettings->setMode(Argus::EGL_STREAM_MODE_MAILBOX);
  if (err) {
    ERROR << "producer:Could not set EGL_STREAM_MODE_MAILBOX";
    return false;
  }

  // err = _isettings->setFifoLength(_fifo_length);
  // if (err) {
  //   ERROR << "producer:Could not set FifoLength to " << _fifo_length << ".";
  //   return false;
  // }

  auto res = _imode->getResolution();
  err = _isettings->setResolution(res);
  DEBUG << "producer:Setting IEGLOutputStream resolution: " << res.width()
        << "x" << res.height();
  if (err) {
    ERROR << "producer:IEGLOutputStreamSettings would not accept resolution: "
          << res.width() << "x" << res.height() << " (status " << err << ").";
    return false;
  }

  _stream.reset(_isession->createOutputStream(_settings.get(), &err));
  if (err) {
    ERROR << "producer:Could not create OutputStream (status " << err << ").";
    return false;
  }

  _istream = Argus::interface_cast<Argus::IEGLOutputStream>(_stream);
  if (!_istream) {
    ERROR << "producer:Could not get IEGLOutputStream from OutputStream.";
    return false;
  }

  // lock request settings
  std::unique_lock<std::mutex> lock(_settings_mx);

  DEBUG << "producer:Creating capture request.";
  // NOTE: manual disables autofocus and awb
  _request.reset(_isession->createRequest(Argus::CAPTURE_INTENT_MANUAL, &err));
  if (err) {
    ERROR << "producer:Could not create capture request (status " << err
          << ").";
    return false;
  }
  _irequest = Argus::interface_cast<Argus::IRequest>(_request);
  if (!_irequest) {
    ERROR << "producer:Could not get IRequest interface from Request.";
    return false;
  }

  DEBUG << "producer:Enabling OutputStream for request.";
  err = _irequest->enableOutputStream(_stream.get());
  if (err) {
    DEBUG << "producer:Could not enable OutputStream for request. (status "
          << err << ").";
    return false;
  }
  _irequest = Argus::interface_cast<Argus::IRequest>(_request);
  if (!_irequest) {
    ERROR << "producer:Could not get IRequest interface from Request.";
    return false;
  }
  _isourcesettings = Argus::interface_cast<Argus::ISourceSettings>(_request);
  if (!_isourcesettings) {
    ERROR << "producer:Could not get ISourceSettings interface from Request.";
    return false;
  }

  DEBUG << "producer:Setting SensorMode on Request.";
  err = _isourcesettings->setSensorMode(_mode);
  if (err) {
    ERROR << "producer:Could not set SensorMode on Request (status " << err
          << ").";
    return false;
  }

  // AutoControlSettings to set camera capture settings
  _iautocontrolsettings = Argus::interface_cast<Argus::IAutoControlSettings>(
      _irequest->getAutoControlSettings());
  if (!_iautocontrolsettings) {
    ERROR << "producer:Could not get IAutoControlSettings interface from "
             "Request.";
    return false;
  }

  // if (_iprovider->supportsExtension(Argus::EXT_BAYER_SHARPNESS_MAP)) {
  //   DEBUG << "producer:Enabling Bayer sharpness map.";
  //   auto bayer_settings =
  //       Argus::interface_cast<Argus::Ext::IBayerSharpnessMapSettings>(_request);
  //   if (!bayer_settings) {
  //     ERROR << "producer:Could not get IBayerSharpnessMapSettings interface "
  //              "from Request (status "
  //           << err << ").";
  //   }
  //   bayer_settings->setBayerSharpnessMapEnable(true);
  // } else {
  //   DEBUG << "producer:Bayer sharpness map not supported";
  // }

  // success
  return true;
}

bool Producer::on_running() {
  auto err = update_request();
  if (err == Argus::STATUS_UNAVAILABLE) {
    DEBUG << "producer:Not yet available. Retrying.";
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    on_running();
  }
  if (err) {
    return false;
  }
  return true;
}

bool Producer::cleanup() {
  DEBUG << "producer:Cleaning up.";

  // reset device
  _device = nullptr;

  // cleanup any session
  if (_isession) {
    DEBUG << "producer:Cancelling CaptureSession Requests.";
    _isession->cancelRequests();
    DEBUG << "producer:Waiting for idle CaptureSession.";
    _isession->waitForIdle();
  }
  _isession = nullptr;
  if (_session) {
    DEBUG << "producer:Resetting CaptureSession to nullptr.";
    _session.reset(nullptr);
  }

  // cleanup capture seettings
  _isettings = nullptr;
  if (_settings) {
    DEBUG << "producer:Resetting OutputStreamSettings to nullptr.";
    _settings.reset(nullptr);
  }

  // cleanup any stream
  if (_istream) {
    DEBUG << "producer:Disconnecting IEGLOutputStream";
    _istream->disconnect();
  }
  _istream = nullptr;
  if (_stream) {
    DEBUG << "producer:Resetting OutputStream to nullptr.";
    _stream.reset(nullptr);
  }

  // cleanup any request and settings
  std::unique_lock<std::mutex> lock(_settings_mx);
  _mode = nullptr;
  _imode = nullptr;
  _irequest = nullptr;
  _isourcesettings = nullptr;
  _iautocontrolsettings = nullptr;
  if (_request) {
    DEBUG << "producer:Resetting Request to nullptr.";
    _request.reset(nullptr);
  }
  lock.unlock();

  // cleanup any camera provider and interface
  if (_provider) {
    DEBUG << "producer:Resetting CameraProvider to nullptr.";
    _provider.reset(nullptr);
  }
  _iprovider = nullptr;

  DEBUG << "producer:Cleanup done. Ready for start.";
  return true;
}

Argus::ICameraProperties* Producer::get_properties() {
  if (!_device) {
    ERROR << "producer:No device to get properties from.";
    return nullptr;
  }

  auto properties = Argus::interface_cast<Argus::ICameraProperties>(_device);
  if (!properties) {
    ERROR << "producer:Could not get ICameraProperties interface from _device.";
    return nullptr;
  }

  return properties;
}

std::vector<Argus::SensorMode*> Producer::get_modes() {
  auto modes = std::vector<Argus::SensorMode*>();

  auto properties = get_properties();
  if (!properties) {
    return modes;
  }

  if (Argus::STATUS_OK != properties->getAllSensorModes(&modes)) {
    ERROR << "producer:Could not get sensor modes from csi id " << _csi_id
          << ".";
    return modes;
  }

  if (modes.size() == 0) {
    ERROR << "producer:No sensor modes are available for csi id " << _csi_id
          << ".";
    return modes;
  }

  return modes;
}

bool Producer::set_mode(Argus::SensorMode* mode) {
  Argus::Status err;

  auto imode = Argus::interface_cast<Argus::ISensorMode>(mode);
  if (!imode) {
    ERROR << "producer:Could not get ISensorMode interface from mode.";
    return false;
  }

  auto res = imode->getResolution();
  err = _isettings->setResolution(res);
  if (err) {
    ERROR << "producer:IEGLOutputStreamSettings would not accept resolution: "
          << res.width() << "x" << res.height() << " (status " << err << ").";
    return false;
  }

  DEBUG << "producer:Setting SensorMode on Request.";
  err = _isourcesettings->setSensorMode(mode);
  if (err) {
    ERROR << "producer:Could not set SensorMode on Request (status " << err
          << ").";
    return false;
  }

  _mode = mode;
  _imode = imode;

  return true;
}

bool Producer::set_mode(uint32_t csi_mode) {
  DEBUG << "producer:Setting csi mode " << csi_mode << ".";
  auto modes = get_modes();
  if (modes.empty()) {
    return false;
  }

  if (modes.size() <= csi_mode) {
    ERROR << "producer:Requested csi_mode does not exist. Valid modes: 0 to "
          << modes.size() - 1 << ".";
    return false;
  }

  return set_mode(modes[csi_mode]);
}

Argus::Status Producer::update_request() {
  DEBUG << "producer:Starting/updating repeat capture.";
  auto err = _isession->repeat(_request.get());
  if (err) {
    ERROR << "producer:Could not start/update repeat capture (status " << err
          << ").";
  }
  return err;
}

Argus::ISensorMode* Producer::get_imode() {
  if (!_imode) {
    ERROR << "producer:Could not get mode since no mode is yet set.";
    return nullptr;
  }
  return _imode;
}

bool Producer::get_resolution(Argus::Size2D<uint32_t>& out) {
  if (!_imode) {
    ERROR << "producer:Could not get resolution since no mode is set yet";
    return false;
  }
  out = _imode->getResolution();
  return true;
}

Argus::Status Producer::set_exposure_time_range(Argus::Range<uint64_t> range) {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_request && _isourcesettings)) {
    return Argus::Status::STATUS_UNAVAILABLE;
  }
  auto err = _isourcesettings->setExposureTimeRange(range);
  if (err) {
    return err;
  }
  return update_request();
}

std::experimental::optional<Argus::Range<uint64_t>>
Producer::get_exposure_time_range() {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_request && _isourcesettings)) {
    ERROR << "producer:Could not get exposure time range.";
    return std::experimental::nullopt;
  }
  return _isourcesettings->getExposureTimeRange();
}

std::experimental::optional<Argus::Range<uint64_t>>
Producer::get_supported_exposure_time_range() {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_mode && _imode)) {
    ERROR << "producer:Could not get supported exposure time range.";
    return std::experimental::nullopt;
  }
  return _imode->getExposureTimeRange();
}

std::experimental::optional<Argus::Range<uint64_t>>
Producer::get_supported_frame_duration_range() {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_mode && _imode)) {
    ERROR << "producer:Could not get supported frame duration range.";
    return std::experimental::nullopt;
  }
  return _imode->getFrameDurationRange();
}

Argus::Status Producer::set_analog_gain_range(Argus::Range<float> range) {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_request && _isourcesettings)) {
    return Argus::Status::STATUS_UNAVAILABLE;
  }
  auto err = _isourcesettings->setGainRange(range);
  if (err) {
    return err;
  }
  return update_request();
}

std::experimental::optional<Argus::Range<float>>
Producer::get_analog_gain_range() {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_request && _isourcesettings)) {
    return std::experimental::nullopt;
  }
  return _isourcesettings->getGainRange();
}

std::experimental::optional<Argus::Range<float>>
Producer::get_supported_analog_gain_range() {
  std::unique_lock<std::mutex> lock(_settings_mx);
  if (!(_mode && _imode)) {
    return std::experimental::nullopt;
  }
  return _imode->getAnalogGainRange();
}

Argus::OutputStream* Producer::get_output_stream() {
  if (!_stream) {
    ERROR << "producer:Could not get OutputStream since not yet created.";
    return nullptr;
  }
  return _stream.get();
}

Producer::~Producer() {
  DEBUG << "producer:Destructor reached.";
}

}  // namespace nvcvcam