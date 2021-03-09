/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef E7978329_D57B_498A_A379_7CA3422BD4E6
#define E7978329_D57B_498A_A379_7CA3422BD4E6

namespace nvcvcam {

// LibArgus currently supports and recommends using PIXEL_FMT_YCbCr_420_888 or
// PIXEL_FMT_P016 while setting the pixelformat in OutputStreamSettings object.
// source:
// https://www.e-consystems.com/Articles/Camera/detailed_guide_to_libargus_with_surveilsquad.asp

// un-helpfully not in the headers themselves. Bayer also works.

/**
 * @brief Formats supported (kinda) by both Argus and OpenCV.
 */
enum class Format {
  BAYER,
  // BGRA, // TODO(BGRA conversion using ISP?)
  Y16,     // 16 bit greyscale (uses P016 internally, maps Y plane)
  Y8,      // 8 bit greyscale (uses YUV420 internally, maps Y plane)
  YUV420,  // 888
  NV12,    // same as YUV420
  P016,    // 16 bit Planar 4:2:0 YUV (broken in argus)
};

}  // namespace nvcvcam

#endif /* E7978329_D57B_498A_A379_7CA3422BD4E6 */
