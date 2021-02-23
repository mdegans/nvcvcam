/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * mdegans wuz here - adapted from cudaBayerDemosaic mmapi sample
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

#ifndef BC3C4619_AD73_4F51_BF00_F761D7D3D5C0
#define BC3C4619_AD73_4F51_BF00_F761D7D3D5C0

#include <Argus/Argus.h>
#include <cuda.h>
#include <cudaEGL.h>

namespace nvcvcam::utils {

bool printCUDAEGLFrame(const CUeglFrame& cudaEGLFrame);

/**
 * @brief Initializes cuda and creates a context.
 *
 * @param ctx a pointer to a cuda context
 * @return true on success
 * @return false on failure
 */
bool init_cuda(CUcontext* ctx);

/**
 * @brief Get the Camera Device object from a CameraProvider.
 *
 * @param cameraProvider
 * @param cameraDeviceIndex
 *
 * @return Argus::CameraDevice*
 */
Argus::CameraDevice* getCameraDevice(Argus::ICameraProvider* iProvider,
                                     uint32_t csi_id);

/**
 * @brief Get the SensorMode from a CameraDevice.
 *
 * @param cameraDevice device to get the sensor mode from
 * @param csi_mode requested sensor mode
 *
 * @return Argus::SensorMode*
 */
Argus::SensorMode* getSensorMode(Argus::CameraDevice* cameraDevice,
                                 uint32_t csi_mode);

}  // namespace nvcvcam::utils
#endif /* BC3C4619_AD73_4F51_BF00_F761D7D3D5C0 */
