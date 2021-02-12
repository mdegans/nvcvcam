/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "demosaic_kernel.hpp"
#include <stdio.h>

#ifndef CU_EGL_COLOR_FORMAT_BAYER_RGGB
#define CU_EGL_COLOR_FORMAT_BAYER_RGGB (0x2D)
#define CU_EGL_COLOR_FORMAT_BAYER_BGGR (0x2E)
#define CU_EGL_COLOR_FORMAT_BAYER_GRBG (0x2F)
#define CU_EGL_COLOR_FORMAT_BAYER_GBRG (0x30)
#endif

// Constant used to store the component ordering of the Bayer input (used by bayerToRgba).
// These values provide the indexes into the original data that will provide an RGGB ordering.
__constant__ short bayerPattern[4];

// Converts a 16-bit Bayer quad to 32bit BGRA. The Bayer components are provided
// in the order they're stored in the buffer, as this function will also handle
// the component ordering during conversion using the 'bayerPattern' constant.
static __device__ uchar4
bayerToRgba(short bayerQuad[4])
{
    // Signed 16-bit Bayer maps 1<<14 to white.
    const float whitePoint = 1<<14;

    // Order the Bayer components based on the format component ordering.
    short r  = bayerQuad[bayerPattern[0]];
    short g1 = bayerQuad[bayerPattern[1]];
    short g2 = bayerQuad[bayerPattern[2]];
    short b  = bayerQuad[bayerPattern[3]];

    // Map [0, 1<<14] to [0, 255].
    uchar4 bgra;
    bgra.z = ((float)b / whitePoint) * 255;
    bgra.y = ((float)((g1 + g2) / 2) / whitePoint) * 255;
    bgra.x = ((float)r / whitePoint) * 255;

    return bgra;
}

// Demosaics a Bayer buffer into an BGRA output.
__global__ void
bayerDemosaicKernel(short* bayerSrc,
                    int bayerWidth,
                    int bayerHeight,
                    int bayerPitch,
                    uchar4* bgraDst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int stepX = blockDim.x * gridDim.x;
    int stepY = blockDim.y * gridDim.y;

    int bgraWidth = bayerWidth / 2;
    int bgraHeight = bayerHeight / 2;

    for (int col = x; col < bgraWidth; col += stepX)
    {
        for (int row = y; row < bgraHeight; row += stepY)
        {
            // Extract the Bayer quad.
            short* bayerOffset = bayerSrc + (col * 2) + (row * bayerPitch);
            short bayerQuad[4];
            bayerQuad[0] = *(bayerOffset);
            bayerQuad[1] = *(bayerOffset + 1);
            bayerQuad[2] = *(bayerOffset + (bayerPitch / 2));
            bayerQuad[3] = *(bayerOffset + (bayerPitch / 2) + 1);

            // Demosaic the Bayer quad to BGRA.
            uchar4 bgra = bayerToRgba(bayerQuad);

            // Optional: Add gain (useful to debug dark output).
            unsigned int gain = 5;
            bgra.x = umin(bgra.x * gain, 255);
            bgra.y = umin(bgra.y * gain, 255);
            bgra.z = umin(bgra.z * gain, 255);

            // Output the BGRA pixels to the output buffer.
            bgraDst[bgraWidth * row + col] = bgra;
        }
    }
}

// Sets the Bayer pattern constant used to order Bayer components.
static void setBayerPatternConstant(int bayerFormat)
{
    short pattern[4];
    if (bayerFormat == CU_EGL_COLOR_FORMAT_BAYER_RGGB)
    {
        pattern[0] = 0;
        pattern[1] = 1;
        pattern[2] = 2;
        pattern[3] = 3;
    }
    else if (bayerFormat == CU_EGL_COLOR_FORMAT_BAYER_BGGR)
    {
        pattern[0] = 3;
        pattern[1] = 1;
        pattern[2] = 2;
        pattern[3] = 0;
    }
    else if (bayerFormat == CU_EGL_COLOR_FORMAT_BAYER_GRBG)
    {
        pattern[0] = 1;
        pattern[1] = 0;
        pattern[2] = 3;
        pattern[3] = 2;
    }
    else // bayerFormat == CU_EGL_COLOR_FORMAT_BAYER_GBRG
    {
        pattern[0] = 2;
        pattern[1] = 0;
        pattern[2] = 3;
        pattern[3] = 1;
    }
    cudaMemcpyToSymbol(bayerPattern, pattern, sizeof(pattern));
}

// Entrypoint to the CUDA Bayer Demosaic.
int cudaBayerDemosaic(CUdeviceptr bayerSrc,
                      int bayerWidth,
                      int bayerHeight,
                      int bayerPitch,
                      int bayerFormat,
                      cudaStream_t stream,
                      CUdeviceptr bgraDst)
{
    setBayerPatternConstant(bayerFormat);

    dim3 threadsPerBlock(32, 32);
    dim3 blocks(2, 2);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    bayerDemosaicKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            (short*)bayerSrc, bayerWidth, bayerHeight, bayerPitch, (uchar4*)bgraDst);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedMillis;
    cudaEventElapsedTime(&elapsedMillis, start, stop);

    return 0;
}
