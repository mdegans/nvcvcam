/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#include "stoppable_thread.hpp"

#include "nvcvcam_error.hpp"

#include <chrono>
#include <exception>

namespace nvcvcam::thread {

bool StoppableThread::start(bool block, std::chrono::nanoseconds timeout) {
  DEBUG << "spawning thread";
  _thread.reset(new std::thread(&StoppableThread::execute, this));

  if (block) {
    return wait(State::RUNNING, timeout);
  }

  return true;
}

void StoppableThread::execute() {
  // setup the thread
  _state = State::INITIALIZING;
  if (!setup()) {
    ERROR << "thread setup failed";
    _state = State::FAILED;
  };

  // run the main loop for this thread
  if (_state != State::FAILED) {
    _state = State::RUNNING;
    while (!_stopping && _state == State::RUNNING) {
      if (!tick()) {
        DEBUG << "tick returned false. setting `FAILED` state.";
        _state = State::FAILED;
        break;
      };
      std::this_thread::yield();
    }
  }

  if (!cleanup()) {
    // Terminate here rather than leak resources.
    _state = State::FAILED;
    ERROR << "cleanup failed. fatal. terminating.";
    std::terminate();
  }

  _state = State::STOPPED;
  _stopping = false;
}

bool StoppableThread::stop(bool block, std::chrono::nanoseconds timeout) {
  DEBUG << "requesting stop";
  if (_thread) {
    _stopping = true;
    if (block) {
      bool success = wait(State::STOPPED, timeout);
      DEBUG << "joining thread";
      _thread->join();
      _thread.reset();
      return success;
    }
  }
  return true;
}

bool StoppableThread::wait(State state, std::chrono::nanoseconds timeout) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // wait for the chosen state
  DEBUG << "waiting for state: " << state << " for " << timeout.count() << "ns";
  while (_state != state) {
    auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    if (elapsed > timeout) {
      ERROR << "timed out waiting for thread RUNNING state";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(SLEEP_INTERVAL_NS));
  }

  return true;
}

StoppableThread::~StoppableThread() {
  DEBUG << "Destructor reached. Stopping.";
  stop();
}

};  // namespace nvcvcam::thread