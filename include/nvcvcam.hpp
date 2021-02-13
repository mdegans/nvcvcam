/* nvcvcam.hpp -- NvCvCam
 *
 * Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef B7D7826A_3396_4C10_A39A_9CC7D54E3972
#define B7D7826A_3396_4C10_A39A_9CC7D54E3972

#include "nvcvcam_config.hpp"

#include <Argus/Argus.h>

#include <opencv2/core/cuda.hpp>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

namespace nvcvcam {

class NvCvCam {
  /** Csi id of the camera */
  uint32_t csi_id;
  /** Argus mode of the camera */
  uint32_t csi_mode;

  /** Whether close has been called */
  std::atomic_bool stopping;

  /** The current status of the producer */
  std::atomic<Argus::Status> producer_status;
  /** The current status of the consumer */
  std::atomic<Argus::Status> consumer_status;
  /** an EGL stream interface to consume frames from */
  Argus::UniqueObj<Argus::OutputStream> raw_stream;
  /** produces frames for the EGL stream */
  std::unique_ptr<std::thread> producer_thread;
  /** consumes frames from the EGL stream*/
  std::unique_ptr<std::thread> consumer_thread;
  /** notifies when the raw_stream is ready */
  std::condition_variable stream_ready;
  /** notifies when shutdown has been requested */
  std::condition_variable shutdown_requested;
  /** notifies when both threads and a frame are ready */
  std::condition_variable capture_ready;

  /** latest GpuMat produced and converted */
  cv::cuda::GpuMat latest_mat;
  /** lock for the GpuMat */
  std::mutex mat_lock;

  /** function to produce frames */
  void producer();
  /** function to convert frames */
  void consumer();

 public:
  NvCvCam()
      : csi_id(defaults::CSI_ID),
        csi_mode(defaults::CSI_MODE),
        stopping(false),
        producer_status(Argus::Status::STATUS_DISCONNECTED),
        consumer_status(Argus::Status::STATUS_DISCONNECTED),
        raw_stream(nullptr),
        producer_thread(nullptr),
        consumer_thread(nullptr){};
  virtual ~NvCvCam() = default;

  /**
   * @brief Open or re-open a camera.
   *
   * @param csi_id the CSI id of the camera.
   * @param csi_mode the Argus CSI mode to request.
   * @return true on success
   * @return false on failure
   */
  virtual bool open(uint32_t csi_id = defaults::CSI_ID,
                    uint32_t csi_mode = defaults::CSI_MODE);

  virtual bool close();

  /**
   * @brief get the latest frame.
   *
   * @return true on success.
   * @return false on failure.
   */
  virtual bool read(cv::cuda::GpuMat& out);
};

}  // namespace nvcvcam

#endif /* B7D7826A_3396_4C10_A39A_9CC7D54E3972 */
