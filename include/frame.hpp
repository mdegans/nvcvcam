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

  cv::cuda::GpuMat& gpu_mat() { return _mat; }
  bool get_debayered(cv::cuda::GpuMat& out,
                     const DebayerGains& gains,
                     cv::cuda::Stream& stream = cv::cuda::Stream::Null());

 private:
  CUgraphicsResource _resource;
  CUeglStreamConnection _conn;
  cudaStream_t _stream;
  CUeglFrame _raw_frame;
  cv::cuda::GpuMat _mat;

  bool sync();
};

}  // namespace nvcvcam
#endif /* DD21561A_FBDD_4A5F_A71F_D6ACC829E542 */
