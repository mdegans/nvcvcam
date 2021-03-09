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

#define LOG INFO << TESTNAME << ":"
static const uint64_t MAX_EXP = 500000000;  // 1/2 second in ns
static const uint64_t EXP_STEP = 1.0 / 60.0 * 1000000000.0;

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
  LOG << ":starting";

  LOG << ":creating camera";
  nvcvcam::NvCvCam camera;

  cv::cuda::GpuMat mat_a;  // debayered
  cv::cuda::GpuMat mat_b;  // scaled
  cv::cuda::GpuMat mat_c;  // 8 bit
  cv::Mat showme(cv::Size(640, 480), CV_8UC4);

  LOG << ":opening camera";
  assert(camera.open());

  // test our `supported` getters
  auto exp_range = camera.get_supported_exposure().value();
  uint64 exp = exp_range.min();
  LOG << "minimum exposure time:" << exp_range.min();
  LOG << "maximum exposure time:" << exp_range.max();
  auto gain_range = camera.get_supported_analog_gain().value();
  float gain = gain_range.min();
  LOG << "minimum gain:" << gain_range.min();
  LOG << "maximum gain:" << gain_range.max();

  LOG << "beginning captures. press esc to stop.";
  do {
    // set parameters for this frame capture
    LOG << "setting exposure time to " << exp;
    assert(camera.set_exposure(exp));
    auto actual_exp = camera.get_exposure().value();
    LOG << "actual exposure set: min:" << actual_exp.min()
        << " max:" << actual_exp.max();
    assert(actual_exp.min() == actual_exp.max());

    LOG << "setting analog gain to " << gain;
    assert(camera.set_analog_gain(gain));
    auto actual_gain = camera.get_analog_gain().value();
    LOG << "actual gain set: min:" << actual_gain.min()
        << " max:" << actual_gain.max();
    assert(actual_gain.min() == actual_gain.max());

    auto frame = camera.capture();
    assert(frame);

    INFO << TESTNAME << ":debayering frame";
    cv::cuda::demosaicing(frame->gpu_mat(), mat_a, cv::COLOR_BayerRG2BGRA, 4);

    LOG << "scaling to 640x480";
    cv::cuda::resize(mat_a, mat_b, cv::Size(640, 480));

    // optional gamma correction / lut / whatever here

    LOG << "converting to 8 bit";
    // https://answers.opencv.org/question/207313/conversion-16bit-image-to-8-bit-image/
    // see note: The factor is not 1/256 but 1/257 because you map range
    // (0-65535) to (0-255), 65535/255 = 257. This is a common off-by-one error
    // in range mapping.
    mat_b.convertTo(mat_c, CV_8UC4, 1.0 / 257.0);

    // NOTE(mdegans): while the imshow docs claims you can, it does not appear
    //  GpuMat is supported in imshow directly because:
    //  "error: (-213:The function/feature is not implemented) You should
    //  explicitly call download method for cuda::GpuMat object in function
    //  'getMat_'"
    LOG << "downloading frame";
    mat_c.download(showme);

    cv::imshow("capture view, press esc to exit", showme);

    // increment set values
    gain = (gain + 1.0) > gain_range.max() ? gain_range.min() : gain + 1.0;
    exp = (exp + EXP_STEP) > exp_range.max() ? exp_range.min() : exp + EXP_STEP;

  } while (27 != cv::waitKey(1));

  LOG << "closing camera";
  assert(camera.close());

  INFO << "exiting test " << TESTNAME;
  return 0;
}