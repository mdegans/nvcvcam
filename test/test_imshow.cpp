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

#include <assert.h>

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
  INFO << TESTNAME << ":starting";

  INFO << TESTNAME << ":creating camera";
  nvcvcam::NvCvCam camera;

  cv::cuda::GpuMat mat_a;  // debayered
  cv::cuda::GpuMat mat_b;  // scaled
  cv::cuda::GpuMat mat_c;  // 8 bit
  cv::Mat showme(cv::Size(640, 480), CV_8UC4);

  INFO << TESTNAME << ":opening camera";
  assert(camera.open());

  INFO << TESTNAME << ":beginning captures. press esc to stop.";
  do {
    INFO << TESTNAME << ":capturing frame";
    auto frame = camera.capture();
    assert(frame);

    INFO << TESTNAME << ":getting debayered frame";
    assert(frame->get_debayered(mat_a));

    INFO << TESTNAME << ":scaling to 640x480";
    cv::cuda::resize(mat_a, mat_b, cv::Size(640, 480));

    INFO << TESTNAME << ":converting to 8 bit";
    // https://answers.opencv.org/question/207313/conversion-16bit-image-to-8-bit-image/
    // see note: The factor is not 1/256 but 1/257 because you map range
    // (0-65535) to (0-255), 65535/255 = 257. This is a common off-by-one error
    // in range mapping.
    mat_b.convertTo(mat_c, CV_8UC4, 1.0 / 257.0);

    // optional gamma correction / lut / whatever here

    // NOTE(mdegans): while the imshow docs claims you can, it does not appear
    //  GpuMat is supported in imshow directly because:
    //  "error: (-213:The function/feature is not implemented) You should
    //  explicitly call download method for cuda::GpuMat object in function
    //  'getMat_'"
    INFO << TESTNAME << ":downloading frame";
    mat_c.download(showme);

    cv::imshow("capture view, press esc to exit", showme);
  } while (27 != cv::waitKey(1));

  INFO << TESTNAME << ":closing camera";
  assert(camera.close());

  INFO << "exiting test " << TESTNAME;
  return 0;
}