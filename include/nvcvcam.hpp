/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef B7D7826A_3396_4C10_A39A_9CC7D54E3972
#define B7D7826A_3396_4C10_A39A_9CC7D54E3972

#include "format.hpp"
#include "frame.hpp"
#include "nvcvcam_config.hpp"
#include "producer.hpp"

#include <cudaEGL.h>
#include <opencv2/core/cuda.hpp>

#include <atomic>
#include <memory>

namespace nvcvcam {

class NvCvCam {
 public:
  NvCvCam() = default;
  NvCvCam(NvCvCam&& other) = default;

  virtual ~NvCvCam();

  NvCvCam& operator=(NvCvCam&& other) = default;

  /**
   * @brief Open or re-open a camera.
   *
   * @param csi_id the CSI id of the camera.
   * @param csi_mode the Argus CSI mode to request.
   * @param format to capture in
   *
   * @return true on success
   * @return false on failure
   */
  virtual bool open(uint32_t csi_id = defaults::CSI_ID,
                    uint32_t csi_mode = defaults::CSI_MODE,
                    Format format = Format::BAYER);

  /**
   * @brief Close the camera and free any resources.
   *
   * NOTE: called automatically by destructor.
   *
   * @return true
   * @return false
   */
  virtual bool close();

  /**
   * @brief Get the latest frame.
   *
   * @return a unique Frame on success.
   * @return nullptr on failure.
   */
  std::unique_ptr<Frame> capture();

  /**
   * @brief Check camera status.
   *
   * @return true if camera is ready for capture
   * @return false if camera is not yet ready
   */
  virtual bool ready();

  /**
   * @brief Set the exposure in nanoseconds.
   *
   * @param ns desired exposure time in nanoseconds. 0 turns AE on.
   *
   * @return true on success
   * @return false on failure
   */
  virtual bool set_exposure(uint64_t ns);

  /**
   * @brief Get the current exposure range. (a range because AE may be on).
   *
   * @return a range on success or std::nullopt on failure
   */
  virtual std::experimental::optional<Argus::Range<uint64_t>> get_exposure();

  /**
   * @brief Get the supported exposure range for this camera and mode.
   *
   * @return a range on success or std::nullopt on failure
   */
  virtual std::experimental::optional<Argus::Range<uint64_t>>
  get_supported_exposure();

  /**
   * @brief Set the analog gain. To check the supported range, use
   * `get_supported_analog_gain`. Negative values turn auto-gain on.
   *
   * @param gain to set
   * @return true on success
   * @return false on failure
   */
  virtual bool set_analog_gain(float gain);

  /**
   * @brief Get the analog gain range. (a range because auto-gain may be on).
   *
   * @return a range on success or std::nullopt on failure
   */
  virtual std::experimental::optional<Argus::Range<float>> get_analog_gain();

  /**
   * @brief Get the supported analog gain range.
   *
   * @return a range on success or std::nullopt on failure
   */
  virtual std::experimental::optional<Argus::Range<float>>
  get_supported_analog_gain();

 private:
  // FIXME(mdegans): ToTW 187 says this is wrong and I should use optional for
  //  delayed initialization:
  //  https://abseil.io/tips/187
  std::unique_ptr<Producer> _producer = nullptr;
  // FIXME(mdegans): Google style guide says trailing underscores are preferred.
  cudaStream_t _cuda_stream = nullptr;
  CUcontext _ctx = nullptr;
  CUeglStreamConnection _cuda_conn = nullptr;
};

}  // namespace nvcvcam

#endif /* B7D7826A_3396_4C10_A39A_9CC7D54E3972 */
