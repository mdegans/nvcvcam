# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

nvcvcam_config_hpp = configure_file(
  input: 'nvcvcam_config.hpp.in',
  output: 'nvcvcam_config.hpp',
  configuration: {
    'PROJECT_VERSION': meson.project_version(),
    'PROJECT_VERSION_MAJOR': proj_ver_major.to_int(),
    'PROJECT_VERSION_MINOR': proj_ver_minor.to_int(),
    'PROJECT_VERSION_PATCH': proj_ver_patch.to_int(),
    'PROJECT_VERSION_GIT': proj_ver_git,
  },
)

nvcvcam_incdir = include_directories('.')

nvcvcam_public_headers = [
  nvcvcam_config_hpp,
  'nvcvcam.hpp',
  'nvcvcam_error.hpp',
  'demosaic_kernel.hpp',
]

install_headers(
  nvcvcam_public_headers,
  subdir: meson.project_name(),
)