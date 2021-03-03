/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef DD21561A_FBDD_4A5F_A71F_D6ACC829E542
#define DD21561A_FBDD_4A5F_A71F_D6ACC829E542

#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>

namespace nvcvcam {

class Frame {
 public:
  Frame(Frame const&) = delete;
  Frame(CUgraphicsResource resource,
        CUeglStreamConnection conn,
        cudaStream_t stream = nullptr);
  virtual ~Frame();

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
   * @brief Get a debayered version of the GpuMat stored internally. Gamma
   * correction is not applied, (this is the raw frame demosaiced, that's it).
   *
   * @param out a GpuMat. Passed to cv::cuda::demosaicing's out parameter.
   * @param code OpenCV color space conversion code. Default RGGB bilinear. See:
   * https://docs.opencv.org/4.5.1/db/d8c/group__cudaimgproc__color.html#ga7fb153572b573ebd2d7610fcbe64166e
   * NOTE: Malvar-He-Cutler is currently doesn't work with 16u.
   * @param stream an optional CUDA stream to run the kernel in (non-blocking)
   *
   * @return true on success
   * @return false on failure
   */
  bool get_debayered(cv::cuda::GpuMat& out,
                     int code = cv::COLOR_BayerRG2BGR,
                     cv::cuda::Stream& stream = cv::cuda::Stream::Null());

  /**
   * @return resolution of the frame.
   */
  cv::Size size();

 private:
  std::atomic_bool _synced;
  CUgraphicsResource _resource;
  CUeglStreamConnection _conn;
  cudaStream_t _stream;
  CUeglFrame _raw_frame;
  cv::cuda::GpuMat _mat;

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
