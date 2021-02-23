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

bool StoppableThread::run() {
  if (_state == State::FAILED) {
    ERROR << "stoppable:Cannot run. Thread in failed state.";
    return false;
  }

  _state = State::RUNNING;
  if (!on_running()) {
    ERROR << "stoppable:`on_running` requested abort.";
    _state = State::FAILED;
  }

  while (!_stopping && _state == State::RUNNING) {
    if (!tick()) {
      DEBUG << "stoppable:`tick` returned `false`. setting `FAILED` state.";
      _state = State::FAILED;
      return false;
    };
    std::this_thread::yield();
  }

  if (_state == State::FAILED) {
    return false;
  } else {
    return true;
  }
}

void StoppableThread::execute() {
  // setup the thread
  _state = State::INITIALIZING;
  if (!setup()) {
    ERROR << "stoppable:Thread setup failed";
    _state = State::FAILED;
  };

  // run the main loop for this thread
  if (!run()) {
    DEBUG << "stoppable:Run failed.";
  }

  if (!cleanup()) {
    // Terminate here rather than leak resources.
    _state = State::FAILED;
    ERROR << "stoppable:Cleanup failed. Fatal. Terminating.";
    std::terminate();
  }

  _state = State::STOPPED;
  _stopping = false;
}

bool StoppableThread::stop(bool block, std::chrono::nanoseconds timeout) {
  DEBUG << "stoppable:Requesting stop";
  if (_thread) {
    _stopping = true;
    if (block) {
      bool success = wait(State::STOPPED, timeout);
      DEBUG << "stoppable:Joining thread.";
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
  DEBUG << "stoppable:Waiting for state: " << state << " for "
        << timeout.count() << "ns";
  while (_state != state) {
    auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    if (elapsed > timeout) {
      ERROR << "stoppable:Timed out waiting for thread RUNNING state.";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(SLEEP_INTERVAL_NS));
  }

  return true;
}

StoppableThread::~StoppableThread() {
  DEBUG << "stoppable:Destructor reached. Stopping.";
  stop();
}

};  // namespace nvcvcam::thread