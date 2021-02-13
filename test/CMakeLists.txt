# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

set(TESTS
  test_capture
  test_open_close
)

foreach(test IN LISTS TESTS)
  add_executable("${test}" "${test}.cpp")
  target_compile_definitions("${test}" PRIVATE "LOGFILE=\"${test}.txt\"")
  target_link_libraries("${test}" "${PROJECT_NAME}" Boost::log)
endforeach()
