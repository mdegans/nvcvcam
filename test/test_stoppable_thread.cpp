/* nvcvcam.hpp -- NvCvCam
 *
 * Copyright (C) 2020 Michael de Gans
 *
 * This is a usage example or test and hereby public domain.
 */

#include "stoppable_thread.hpp"

#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <assert.h>

using nvcvcam::thread::State;
using nvcvcam::thread::StoppableThread;

static void setup_logging() {
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

namespace {

class TestStoppable : public StoppableThread {
 protected:
  virtual bool setup();
  virtual bool tick();
  virtual bool cleanup();

 public:
  std::atomic_bool setup_called;
  std::atomic_uint ticks;
  std::atomic_bool cleanup_called;
  std::atomic_bool destructor_called;
  TestStoppable()
      : setup_called(false),
        ticks(0),
        cleanup_called(false),
        destructor_called(false){};
  virtual ~TestStoppable();
};

bool TestStoppable::setup() {
  ticks = 0;
  cleanup_called = false;
  setup_called = true;
  return true;
}

bool TestStoppable::tick() {
  ticks++;
  return true;
}

bool TestStoppable::cleanup() {
  cleanup_called = true;
  return true;
}

TestStoppable::~TestStoppable() {
  BOOST_LOG_TRIVIAL(info) << "TestStoppable destructor called;";
}

}  // namespace

int main() {
  setup_logging();

  BOOST_LOG_TRIVIAL(info) << "starting test " << TESTNAME;

  for (size_t i = 0; i < 2; i++) {
    TestStoppable ts;
    for (size_t i = 0; i < 5; i++) {
      assert(ts.start());
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      assert(ts.setup_called);
      assert(ts.stop());
      assert(ts.ticks);
      assert(ts.cleanup_called);
    }
  }

  BOOST_LOG_TRIVIAL(info) << "exiting test " << TESTNAME;

  return 0;
}