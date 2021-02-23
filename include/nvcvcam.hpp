/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef B7D7826A_3396_4C10_A39A_9CC7D54E3972
#define B7D7826A_3396_4C10_A39A_9CC7D54E3972

#include "frame.hpp"
#include "nvcvcam_config.hpp"
#include "producer.hpp"

#include <cudaEGL.h>
#include <opencv2/core/cuda.hpp>

#include <atomic>
#include <memory>

namespace nvcvcam {

class NvCvCam {
  std::unique_ptr<Producer> _producer;
  cudaStream_t _cuda_stream;
  CUcontext _ctx;
  CUeglStreamConnection _cuda_conn;

 public:
  NvCvCam()
      : _producer(nullptr),
        _cuda_stream(nullptr),
        _ctx(nullptr),
        _cuda_conn(nullptr){};
  NvCvCam(const NvCvCam&) = delete;

  virtual ~NvCvCam() = default;

  NvCvCam& operator=(const NvCvCam&) = delete;

  /**
   * @brief Open or re-open a camera.
   *
   * @param csi_id the CSI id of the camera.
   * @param csi_mode the Argus CSI mode to request.
   *
   * @return true on success
   * @return false on failure
   */
  virtual bool open(uint32_t csi_id = defaults::CSI_ID,
                    uint32_t csi_mode = defaults::CSI_MODE);

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
};

}  // namespace nvcvcam

#endif /* B7D7826A_3396_4C10_A39A_9CC7D54E3972 */
