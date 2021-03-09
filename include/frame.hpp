/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef DD21561A_FBDD_4A5F_A71F_D6ACC829E542
#define DD21561A_FBDD_4A5F_A71F_D6ACC829E542

#include "format.hpp"

#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>

namespace nvcvcam {

class Frame final {
 public:
  Frame(Frame const&) = delete;
  Frame(CUgraphicsResource resource,
        CUeglStreamConnection conn,
        cudaStream_t stream = nullptr,
        Format format = Format::BAYER);
  ~Frame();

  Frame& operator=(Frame const&) = delete;

  /**
   * @brief Get the raw bayer GpuMat owned by Frame. It's lifetime is tied to
   * this frame and should not be used after the Frame is destroyed.
   *
   * NOTE: format is `CV_16UC1`, RGGB order.
   *
   * TODO(mdegans): figure out a way to guarantee this.
   *
   * @return cv::cuda::GpuMat
   */
  cv::cuda::GpuMat gpu_mat();

  /**
   * @brief Format of this frame.
   */
  const Format format;

 private:
  CUgraphicsResource _resource = nullptr;
  CUeglStreamConnection _conn = nullptr;
  cudaStream_t _stream = nullptr;
  std::atomic_bool _synced = ATOMIC_VAR_INIT(false);
  CUeglFrame _raw_frame = CUeglFrame();
  cv::cuda::GpuMat _mat = cv::cuda::GpuMat();

  /**
   * @brief Lazy sync of `_stream`. Only performed when `_mat` is accessed.
   *
   * @return true on success
   * @return false on failure to sync
   */
  bool sync();
};

}  // namespace nvcvcam

#endif /* DD21561A_FBDD_4A5F_A71F_D6ACC829E542 */
