# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

# this meson.build is for gst-check tests (mostly integration tests)

# a list of dicts describing tests
tests = [
  {
    'description': 'test csi camera open close',
    'filename': 'test_open_close',
    'sources': ['test_open_close.cpp'],
    'suites': ['integration'],
  },
  {
    'description': 'test csi capture',
    'filename': 'test_capture',
    'sources': ['test_capture.cpp'],
    'suites': ['integration'],
  },
  {
    'description': 'test imshow',
    'filename': 'test_imshow',
    'sources': ['test_imshow.cpp'],
    'suites': ['integration'],
  },
  {
    'description': 'test stoppable thread',
    'filename': 'test_stoppable_thread',
    'sources': ['test_stoppable_thread.cpp'],
    'suites': ['unit'],
  },
]

# build and run tests (on ninja test)
foreach t: tests
  exe = executable(t['filename'], t['sources'],
    link_with: libnvcvcam,
    dependencies: nvcvcam_deps,
    cpp_args: [
      '-DLOGFILE="' + t['filename'] + '.txt"',
      '-DTESTNAME="' + t['filename'] + '"',
      '-DBOOST_LOG_DYN_LINK=1',
    ],
    include_directories: nvcvcam_incdir,
  )
  # skip running test if it's an interactive one
  if 'interactive' in t['suites']
    continue
  endif
  # run the rest of the tests in parallel
  test(t['description'], exe,
    protocol: 'exitcode',
    is_parallel: 'integration' not in t['suites'],
    suite: t['suites'],
  )
endforeach