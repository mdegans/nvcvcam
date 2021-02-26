#ifndef DFFB9513_C1EE_4EAD_AD16_80CC4C1A3D30
#define DFFB9513_C1EE_4EAD_AD16_80CC4C1A3D30
# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

cmake_module_path = meson.current_source_dir() / '..' / 'cmake'

# dependencies
nvcvcam_deps = [
  # FIXME(mdegans): find a way to check for cuda components
  dependency('opencv4',
    modules: ['cuda'],
  ),
  dependency('Argus',  
    cmake_module_path: cmake_module_path,
  ),
  dependency('EGL',
    cmake_module_path: cmake_module_path,
  ),
  dependency('NVMMAPI',
    cmake_module_path: cmake_module_path,
  ),
  # dependency('OpenGL',
  #   cmake_module_path: cmake_module_path,
  # ),
  # dependency('OpenGLES',
  #   cmake_module_path: cmake_module_path,
  # ),
  dependency('threads'),
  # dependency('cuda',
  #   modules: ['cudart', 'nppicc', 'nppc', 'nppisu', 'nppidei']
  # ),
  dependency('cuda'),
  dependency('boost',
    # unlike CMake, 'thread' and 'system' are not implicit, apparently
    modules: ['log', 'log_setup', 'thread', 'system'],
  ),
# cuEGLStreamConsumerReleaseFrame /usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1 
# cuEGLStreamConsumerReleaseFrame /usr/lib/aarch64-linux-gnu/tegra/libnvofsdk.so 
  cc.find_library('cuda'),
]

nvcvcam_src = [
  'frame.cpp',
  'nvcvcam_error.cpp',
  'nvcvcam.cpp',
  'producer.cpp',
  'stoppable_thread.cpp',
  'utils.cpp',
]

libnvcvcam_libname = meson.project_name() + '-' + proj_ver_short
libnvcvcam = library(libnvcvcam_libname, nvcvcam_src,
  dependencies: nvcvcam_deps,
  cpp_args: [
    '-fno-exceptions',
    '-DBOOST_LOG_DYN_LINK=1',
  ],
  include_directories: nvcvcam_incdir,
)

# use this variable in superprojects
nvcvcam_dep = declare_dependency(
  link_with: libnvcvcam,
  include_directories: nvcvcam_incdir,
  dependencies: nvcvcam_deps,
)

pkg = import('pkgconfig')
nvcvcam_pc = pkg.generate(libnvcvcam,
  description: project_description,
  libraries: nvcvcam_deps,
  url: project_url,
  install_dir: get_option('datadir') / 'pkgconfig',
)
#endif /* DFFB9513_C1EE_4EAD_AD16_80CC4C1A3D30 */
