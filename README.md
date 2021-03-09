# NvCvCam

Is similar to `cv::VideoCapture`, only it's designed specifically for Jetson and
captures frames directly to `cv::GpuMat` for further processing with OpenCV's
CUDA module. It's designed to hide the complexity of using Argus directly and
avoid any unnecessary conversions through a GStreamer pipeline.

## Features:
* Raw, Bayer format capture
* YUV Capture (12 and 16 bit)
* stream support for non-blocking operations where possible

## Requirements
* A Jetson board.
* MMAPI samples (`sudo apt-get install nvidia-l4t-jetson-multimedia-api`)
* [OpenCV with CUDA support for Tegra / Jetson](https://github.com/mdegans/nano_build_opencv).
* Meson (`pip3 install meson`)
* Ninja (`sudo apt-get install ninja-build`)
* boost log (`sudo apt-get install libboost-log-dev`)

## Building
```
meson build
cd build
ninja
```
(and to install)
```
sudo ninja install
```

## Example Use

In your build system, include the `nvcvcam-0.0` pkg-config package. In Meson,
This is `dependency('nvcvcam-0.0')`. The `nvcvcam_dep` variable may also be
imported when this project is used as a subproject. It is not necessary to
install the library when used as a subproject.

With CMake, you could use [FindPkgConfig](https://cmake.org/cmake/help/latest/module/FindPkgConfig.html).

```C++
#include <nvcvcam/nvcvcam.hpp>

int main() {
  auto camera = nvcvcam::NvCvCam;

  cv::cuda::GpuMat debayered;  // this is reused

  assert(camera.open())  // CSI id, CSI mode, and image format may be supplied

  while (auto frame = camera.capture()) {
    // Raw bayer mapped from the Argus EGL stream. It can be modified but **must
    // not outlive the frame**, since the frame is also mapped. No copies here.
    auto raw = frame->gpu_mat();

    // Do things with frame (eg. demosaic, gamma correct, apply a lut,
    // download to cpu, imshow it, twist it, bop it, whatever).

  }  // `frame` is destroyed, and it's resources released here.

  return 0;  // camera.close() is called here, if it hasn't already been.
}
```

## Planned features
* Doxygen docs
* Metadata support, including Bayer sharpness map, and other Argus goodies.
* Rust bindings
* Python bindings