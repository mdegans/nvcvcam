/* Copyright (C) 2020 Michael de Gans
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.mit file for details.
 */

#ifndef E589C439_F4B4_4D67_9114_8A83FA001DB2
#define E589C439_F4B4_4D67_9114_8A83FA001DB2

#include <atomic>
#include <memory>
#include <thread>

namespace nvcvcam::thread {

/** interval for sleeping (eg. while waiting) */
const uint64_t SLEEP_INTERVAL_NS = 100000;

/**
 * Thread states
 */
enum State {
  STOPPED = 0x1,
  INITIALIZING = 0x2,
  RUNNING = 0x4,
  FAILED = 0x8,
};

/**
 * Base class for threads. Derived classes need to `tick` and probably `setup`
 * and `cleanup`.
 */
class StoppableThread {
  /** set to request shutdown of the thread */
  std::atomic_bool _stopping;
  /** state of the thread */
  std::atomic<State> _state;
  /** the actual thread */
  std::unique_ptr<std::thread> _thread;

  /**
   * @brief main loop of the thread. Runs `tick` in a loop while `!_stopping`
   * and `_state` == `State::RUNNING`. Calls cleanup for you on exit.
   *
   * @return true on success
   * @return false on failure
   */
  virtual void execute();

 protected:
  /**
   * @brief setup any resources needed for tick
   *
   * @return true on success
   * @return false on failure
   */
  virtual bool setup() { return true; };
  /**
   * @brief an iteration of the thread loop
   *
   * @return true to continue iteration
   * @return false to stop iteration
   */
  virtual bool tick() = 0;
  /**
   * @brief cleanup any resources needed for tick
   *
   * @return true on success
   * @return false calls terminate
   */
  virtual bool cleanup() { return true; };

 public:
  StoppableThread()
      : _stopping(false), _state(State::STOPPED), _thread(nullptr){};
  virtual ~StoppableThread();

  /**
   * @brief start the thread
   *
   * @param block until ready
   * @param timeout_ns to block for if block
   *
   * @return true on success
   * @return false on failure
   */
  bool start(bool block = true, uint64_t timeout_ns = -1);
  /**
   * @brief wait until a thread state
   *
   * @param state to wait until. `State` can be combined with bitwise operators.
   * @param timeout_ns to wait for
   * @return true
   * @return false
   */
  bool wait(State state, uint64_t timeout_ns = -1);
  /**
   * @brief stop the thread. if already stopped does nothing.
   *
   * NOTE: called by destructor automatically
   *
   * @param block until ready
   * @param timeout_ns to block for if block
   * @return true
   * @return false
   */
  bool stop(bool block = true, uint64_t timeout_ns = -1);
  /**
   * @brief query the state of the thread.
   *
   * @return true if ready to do it's thing
   * @return false if not in the `RUNNING` state
   */
  bool ready() { return _state == State::RUNNING; };
};

}  // namespace nvcvcam::thread
#endif /* E589C439_F4B4_4D67_9114_8A83FA001DB2 */
