# Copyright (C) 2020 Michael de Gans

# This build file may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.mit file for details.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

# Read VERSION file
file(READ "VERSION" VERSION_FILE_CONTENTS)
string(STRIP "${VERSION_FILE_CONTENTS}" VERSION_FILE_CONTENTS)

# get git version
execute_process(COMMAND git rev-parse --short HEAD
  RESULT_VARIABLE GIT_REV_RETCODE
  OUTPUT_VARIABLE PROJECT_VERSION_GIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT GIT_REV_RETCODE EQUAL 0)
  set(PROJECT_VERSION_GIT "untracked")
endif()

# setup project
project(nvcvcam
  VERSION "${VERSION_FILE_CONTENTS}"
  LANGUAGES CXX CUDA
)

# append our .cmake files in case Nvidia's aren't in a sane path
list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/cmake"
)

# we probably don't need all of this, but wtf, the build system will sort it
find_package(Argus REQUIRED COMPONENTS ARGUS_INCLUDE_DIR ARGUS_LIBRARY)
find_package(OpenGLES REQUIRED)
find_package(EGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(NVMMAPI REQUIRED)
# FIXME(mdegans): find a way to check for cuda components
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS log)
find_package(Threads REQUIRED)

# our includes
add_subdirectory(include)
include_directories(PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
include_directories(PUBLIC include)

# our library
add_library("${PROJECT_NAME}" SHARED
  src/demosaic_kernel.cu
  src/nvcvcam.cpp
)

# package includes
target_include_directories("${PROJECT_NAME}"
  PUBLIC
  ${ARGUS_INCLUDE_DIRS}
  ${OPENGLES_INCLUDE_DIR}
  ${EGL_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${NVMMAPI_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# libraries for all targets
target_link_libraries("${PROJECT_NAME}"
  ${ARGUS_LIBRARIES}
  ${OPENGLES_LIBRARIES}
  ${EGL_LIBRARIES}
  ${CMAKE_CUDA_TOOLKIT_LIBRARIES}
  ${NVMMAPI_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  Boost::log
)

# absolute must have tests
add_subdirectory(test)