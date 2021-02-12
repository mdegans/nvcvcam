/* nvcvcam.hpp -- NvCvCam
 *
 * Copyright (C) 2020 Michael de Gans
 *
 * This is a usage example or test and hereby public domain.
 */

#include "nvcvcam_error.hpp"

#include "nvcvcam.hpp"

#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <assert.h>

void setup_logging(const char* logfile) {
  boost::log::add_file_log(logfile);
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::debug);
}

int main() {
  setup_logging(LOGFILE);

  auto camera = nvcvcam::NvCvCam();
  cv::cuda::GpuMat frame;

  assert(camera.open());
  assert(camera.read(frame));
  assert(!frame.empty());
  assert(camera.close());

  return 0;
}