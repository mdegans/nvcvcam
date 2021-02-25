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
#include <opencv2/imgcodecs.hpp>

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
  cv::cuda::GpuMat converted;
  cv::cuda::GpuMat cv_debayered;
  cv::Mat downloaded;
  cv::cuda::GpuMat debayered;

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":opening camera";
  assert(camera.open());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":getting frame...";
  auto frame = camera.capture();

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking frame...";
  assert(frame);

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":getting GpuMat";
  gpumat = frame->gpu_mat();

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking GpuMat";
  assert(!gpumat.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":converting GpuMat";
  gpumat.convertTo(converted, CV_16UC1);

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking converted";
  assert(!converted.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":downloading converted";
  converted.download(downloaded);

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking downloaded";
  assert(!downloaded.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":saving downloaded";
  assert(cv::imwrite("bayer.png", downloaded));

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":OpenCV demosaicing";
  cv::cuda::demosaicing(converted, cv_debayered, cv::COLOR_BayerRG2BGR);

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking cv_debayered";
  assert(!cv_debayered.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":downloading cv_debayered";
  cv_debayered.download(downloaded);

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking downloaded";
  assert(!downloaded.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":saving cv_debayered.png";
  assert(cv::imwrite("cv_debayered.png", downloaded));

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":getting our debayered";
  assert(frame->get_debayered(debayered, nvcvcam::DebayerGains()));

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":downloading debayered";
  debayered.download(downloaded);

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking downloaded";
  assert(!downloaded.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":saving debayered.png";
  assert(cv::imwrite("debayered.png", downloaded));

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":destroying Frame";
  frame.reset();

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":checking debayered";
  assert(!debayered.empty());

  BOOST_LOG_TRIVIAL(info) << TESTNAME << ":closing camera";
  assert(camera.close());

  BOOST_LOG_TRIVIAL(info) << "exiting test " << TESTNAME;
  return 0;
}