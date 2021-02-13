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
  dependency('threads'),
  dependency('cuda',
    modules: ['cudart']
  ),
  dependency('boost',
    # unlike CMake, 'thread' and 'system' are not implicit, apparently
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
  include_directories: nvcvcam_incdir,
)

pkg = import('pkgconfig')
nvcvcam_pc = pkg.generate(libnvcvcam,
  description: project_description,
  libraries: nvcvcam_deps,
  url: project_url,
  install_dir: get_option('datadir') / 'pkgconfig',
)
