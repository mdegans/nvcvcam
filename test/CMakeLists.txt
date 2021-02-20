# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

set(TESTS
  test_capture
  test_open_close
  test_stoppable_thread
)

foreach(test IN LISTS TESTS)
  add_executable("${test}" "${test}.cpp")
  target_include_directories("${PROJECT_NAME}"
    PRIVATE
    ${Boost_INCLUDE_DIR})
  target_compile_definitions("${test}"
    PRIVATE
      "LOGFILE=\"${test}.txt\""
      "TESTNAME=\"${test}\""
      "-DBOOST_LOG_DYN_LINK")
  target_link_libraries("${test}" "${PROJECT_NAME}" ${Boost_LIBRARIES})
endforeach()
