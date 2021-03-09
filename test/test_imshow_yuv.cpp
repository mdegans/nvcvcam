/* nvcvcam.hpp -- NvCvCam
 *
 * Copyright (C) 2020 Michael de Gans
 *
 * This is a usage example or test and hereby public domain.
 */

#include "nvcvcam_error.hpp"

#include "nvcvcam.hpp"

#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>

#define LOG INFO << TESTNAME << ":"

void setup_logging() {
  boost::log::register_simple_formatter_factory<
      boost::log::trivial::severity_level, char>("Severity");
  boost::log::add_file_log(
      LOGFILE, boost::log::keywords::auto_flush = true,
      boost::log::keywords::format = "[%TimeStamp%][%Severity%]: %Message%");
  boost::log::add_console_log(
      std::cout,
      boost::log::keywords::format = "[%TimeStamp%][%Severity%]: %Message%");
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::debug);
  boost::log::add_common_attributes();
}

int main() {
  setup_logging();
  LOG << "starting";

  cv::Size displaysize(cv::Size(640, 480));

  LOG << "creating camera";
  nvcvcam::NvCvCam camera;

  cv::cuda::GpuMat yuv;
  cv::cuda::GpuMat bgr;
  cv::cuda::GpuMat scaled;
  cv::cuda::GpuMat eightbpp;

  cv::Mat showme(displaysize, CV_8UC4);

  LOG << " opening camera in NV12 mode";
  assert(camera.open(0, 0, nvcvcam::Format::NV12));

  LOG << "beginning captures. press esc to stop.";
  do {
    LOG << ":capturing frame";
    auto frame = camera.capture();
    assert(frame);

    yuv = frame->gpu_mat();
    assert(!yuv.empty());

    // LOG << "converting NV12 to BGRA";
    // // cv::COLOR_YUV2BGR_NV12 is unsupported because FML
    // cv::cuda::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR, 4);

    LOG << "scaling to 640x480";
    cv::cuda::resize(yuv, scaled, displaysize);

    LOG << "converting to 8bpp";
    scaled.convertTo(eightbpp, CV_8UC4, 1.0 / 257.0);

    // NOTE(mdegans): while the imshow docs claims you can, it does not appear
    //  GpuMat is supported in imshow directly because:
    //  "error: (-213:The function/feature is not implemented) You should
    //  explicitly call download method for cuda::GpuMat object in function
    //  'getMat_'"
    LOG << "downloading frame";
    scaled.download(showme);

    cv::imshow("capture view, press esc to exit", showme);
  } while (27 != cv::waitKey(1));

  LOG << "closing camera";
  assert(camera.close());

  LOG << "exiting test";
  return 0;
}