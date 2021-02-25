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

  BOOST_LOG_TRIVIAL(info) << "starting test " << TESTNAME;

  nvcvcam::NvCvCam camera;
  cv::cuda::GpuMat gpumat;
  cv::Mat showme;
  nvcvcam::DebayerGains gains{
      1.0f,
      1.0f,
      1.0f,
      1.0f,
  };

  assert(camera.open());
  auto frame = camera.capture();
  assert(frame);
  frame->get_debayered(gpumat, gains);
  gpumat.download(showme);
  cv::imshow("debayered", showme);
  cv::waitKey(0);
  assert(camera.close());

  BOOST_LOG_TRIVIAL(info) << "exiting test " << TESTNAME;
  return 0;
}