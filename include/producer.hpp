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

#ifndef F3BA22E3_6D3F_48E7_81A6_F33CB155167B
#define F3BA22E3_6D3F_48E7_81A6_F33CB155167B

#include "stoppable_thread.hpp"

#include <Argus/Argus.h>

namespace nvcvcam {

class Producer : public thread::StoppableThread {
  uint _csi_id;
  uint _csi_mode;
  uint32_t _fifo_length;

  Argus::UniqueObj<Argus::CameraProvider> _provider;
  Argus::ICameraProvider* _iprovider;
  Argus::CameraDevice* _device;
  Argus::SensorMode* _mode;
  Argus::ISensorMode* _imode;
  Argus::UniqueObj<Argus::CaptureSession> _session;
  Argus::ICaptureSession* _isession;
  Argus::UniqueObj<Argus::OutputStreamSettings> _settings;
  Argus::IEGLOutputStreamSettings* _isettings;
  Argus::UniqueObj<Argus::OutputStream> _stream;
  Argus::IEGLOutputStream* _istream;
  Argus::UniqueObj<Argus::Request> _request;
  Argus::IRequest* _irequest;
  Argus::ISourceSettings* _isourcesettings;

 protected:
  /**
   * @brief Sets up the producer for capture.
   *
   * @return true on success
   * @return false on failure
   */
  virtual bool setup();
  /**
   * @brief This implementation of `tick` enqueues capture requests while the
   * FIFO buffer is not full.
   *
   * @return on success (continues iteration)
   * @return on failure (superclass will set failed status and call cleanup).
   */
  virtual bool tick();
  /**
   * @brief Clean up any camera resources held by the producer.
   *
   * @return true on success
   * @return false on failure
   */
  virtual bool cleanup();

  /**
   * @brief Get the Camera's properites interface.
   *
   * @return Argus::ICameraProperties*
   */
  virtual Argus::ICameraProperties* get_properties();

  /**
   * @brief Set the camera sensor mode on anything needed for capture.
   *
   * NOTE: this will probably be public later
   *
   * @param mode a valid Argus sensor mode (eg. from get_modes()).
   *
   * @return true on success
   * @return false on failure
   */
  bool set_mode(Argus::SensorMode* mode);
  /**
   * @brief Set the camera sensor mode on anything needed for capture.
   *
   * NOTE: this will probably be public later
   *
   * @param mode a valid Argus sensor mode number as uint32_t.
   *
   * @return true on success
   * @return false on failure
   */
  bool set_mode(uint32_t csi_mode);
  /**
   * @brief Request Argus to perform a capture.
   *
   * @param timeout before failure.
   *
   * @return true on success
   * @return false on failure
   */
  bool enqueue_request(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::max());

 public:
  Producer(uint csi_id = 0, uint csi_mode = 0, uint32_t fifo_length = 2)
      : _csi_id(csi_id), _csi_mode(csi_mode), _fifo_length(fifo_length){};
  virtual ~Producer();

  /**
   * @brief Get the current sensor mode interface.
   *
   * @return Argus::ISensorMode*
   * @return nullptr on failure
   */
  Argus::ISensorMode* get_imode();
  /**
   * @brief Get all supported camera modes for the currently selected camera.
   *
   * @return std::vector<Argus::SensorMode*>
   */
  std::vector<Argus::SensorMode*> get_modes();
  /**
   * @brief Get the active resolution.
   *
   * @param out the resolution to set.
   *
   * @return true on success
   * @return false on failure
   */
  bool get_resolution(Argus::Size2D<uint32_t>& out);
  /**
   * @brief Get a pointer to the OutputStream owned by this object. The pointer
   * is valid so long as the producer is `ready()`.
   *
   * @return Argus::OutputStream*
   */
  Argus::OutputStream* get_output_stream();
};

}  // namespace nvcvcam

#endif /* F3BA22E3_6D3F_48E7_81A6_F33CB155167B */
