# nvcam_config.hpp -- configuration for nvcam

# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

# path to libargus .cmake files (change this if Nvidia moves the cmake modules
# to a standard cmake module path)
# mmapi_cmake_path = '/usr/src/jetson_multimedia_api/argus/cmake'

# workaround:
tegra_libdir = '/usr/lib/aarch64-linux-gnu/tegra/'
mmapi_incdirs = [
  include_directories('/usr/src/jetson_multimedia_api/include/'),
  include_directories('/usr/src/jetson_multimedia_api/argus/samples/utils/'),
]

# dependencies
nvcvcam_deps = [
  dependency('opencv4',
    modules: ['cuda'],
  ),
  # FIXME(mdegans): report that this does not seem to be working with current 
  # meson. cmake_module_path is not respected.
  # dependency('Argus',
  #   method: 'cmake',
  #   # cmake_module_path: mmapi_cmake_path,
  #   cmake_args: '-DCMAKE_MODULE_PATH=' + mmapi_cmake_path,
  # ),
  cc.find_library('nvbuf_utils', dirs: tegra_libdir),
  cc.find_library('nvargus', dirs: tegra_libdir),
  dependency('threads'),
  dependency('cuda', modules: ['cudart']),
  dependency('boost',
    modules: ['log', 'thread', 'system'],
  )
]

nvcvcam_src = [
  'nvcvcam.cpp',
  'demosaic_kernel.cu',
]

libnvcvcam_libname = meson.project_name() + '-' + proj_ver_short
libnvcvcam = library(libnvcvcam_libname, nvcvcam_src,
  dependencies: nvcvcam_deps,
  cpp_args: ['-fno-exceptions'],
  include_directories: [nvcvcam_incdir, mmapi_incdirs],
)

pkg = import('pkgconfig')
nvcvcam_pc = pkg.generate(libnvcvcam,
  description: project_description,
  libraries: nvcvcam_deps,
  url: project_url,
  install_dir: get_option('datadir') / 'pkgconfig',
)
