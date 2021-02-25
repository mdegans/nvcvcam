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

#include <atomic>

namespace nvcvcam {

struct DebayerGains {
  float r;
  float g;
  float b;
  float v;
};

class Frame {
 public:
  Frame(Frame const&) = delete;
  Frame(CUgraphicsResource resource,
        CUeglStreamConnection conn,
        cudaStream_t stream = nullptr);
  virtual ~Frame();

  Frame& operator=(Frame const&) = delete;

  /**
   * @brief Get the GpuMat owned by Frame. It's lifetime is tied to this frame
   * and should not be used after the Frame is destroyed.
   *
   * TODO(mdegans): figure out a way to guarantee this.
   *
   * @return cv::cuda::GpuMat
   */
  cv::cuda::GpuMat gpu_mat();
  /**
   * @brief Get a debayered version of the GpuMat stored internally.
   *
   * @param out a GpuMat. If it's not of the same size and type, it will be
   * reallocated (blocking).
   * @param gains RGBV gains for debayering.
   * @param stream an optional CUDA stream to run the kernel in (non-blocking)
   *
   * @return true on success
   * @return false on failure
   */
  bool get_debayered(cv::cuda::GpuMat& out,
                     const DebayerGains& gains,
                     cv::cuda::Stream& stream = cv::cuda::Stream::Null());

 private:
  std::atomic_bool _synced;
  CUgraphicsResource _resource;
  CUeglStreamConnection _conn;
  cudaStream_t _stream;
  CUeglFrame _raw_frame;
  cv::cuda::GpuMat _mat;

  /**
   * @brief Lazy sync of the stream. Only performed when a gpu_mat is first
   * accessed.
   *
   * @return true
   * @return false
   */
  bool sync();
};

}  // namespace nvcvcam
#endif /* DD21561A_FBDD_4A5F_A71F_D6ACC829E542 */
