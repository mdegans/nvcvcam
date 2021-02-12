
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
]

# build and run tests (on ninja test)
foreach t: tests
  exe = executable(t['filename'], t['sources'],
    link_with: libnvcvcam,
    dependencies: nvcvcam_deps,
    cpp_args: [
      '-DLOGFILE="' + t['filename'] + '.txt"',
    ],
    include_directories: [nvcvcam_incdir, mmapi_incdirs],
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