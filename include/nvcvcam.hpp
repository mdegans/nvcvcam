/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef B7D7826A_3396_4C10_A39A_9CC7D54E3972
#define B7D7826A_3396_4C10_A39A_9CC7D54E3972

#include "consumer.hpp"
#include "frame.hpp"
#include "nvcvcam_config.hpp"
#include "producer.hpp"

#include <opencv2/core/cuda.hpp>

#include <atomic>
#include <memory>

namespace nvcvcam {

class NvCvCam {
  std::unique_ptr<Producer> _producer;
  std::unique_ptr<Consumer> _consumer;

 public:
  NvCvCam() : _producer(nullptr), _consumer(nullptr){};
  virtual ~NvCvCam() = default;

  /**
   * @brief open or re-open a camera.
   *
   * @param csi_id the CSI id of the camera.
   * @param csi_mode the Argus CSI mode to request.
   * @param block until open or timeout
   * @param timeout to wait for if block (default waits forever)
   * @return true on success
   * @return false on failure
   */
  virtual bool open(
      uint32_t csi_id = defaults::CSI_ID,
      uint32_t csi_mode = defaults::CSI_MODE,
      bool block = true,
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::max());

  /**
   * @brief close the camera and free any resources.
   *
   * NOTE: called automatically by destructor.
   *
   * @param block until ready
   * @param timeout to block for if block
   *
   * @return true
   * @return false
   */
  virtual bool close(
      bool block = true,
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::max());

  /**
   * @brief request and get the latest frame.
   *
   * @return a unique Frame on success.
   * @return nullptr on failure.
   */
  std::unique_ptr<Frame> capture();
};

}  // namespace nvcvcam

#endif /* B7D7826A_3396_4C10_A39A_9CC7D54E3972 */
