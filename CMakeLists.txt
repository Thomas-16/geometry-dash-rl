cmake_minimum_required(VERSION 3.24)
project(geometry-dash-rl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# This machine's MSYS2 ld (binutils 2.43.1) crashes (exit 116) when linking
# C++ exception unwind info against the shared runtime; fully static linking
# avoids it and also makes the exes run without ucrt64 DLLs on PATH.
if(MINGW)
  add_link_options(-static)
  set(ZLIB_USE_STATIC_LIBS ON)
endif()

find_package(ZLIB REQUIRED)

add_library(gdcore STATIC
  src/core/level.cpp
)
target_include_directories(gdcore PUBLIC src)
target_link_libraries(gdcore PUBLIC ZLIB::ZLIB)

add_executable(dump tools/dump.cpp)
target_link_libraries(dump PRIVATE gdcore)

add_executable(viewer src/viewer/main.cpp src/viewer/atlas.cpp)
target_link_libraries(viewer PRIVATE gdcore)
if(MINGW)
  target_link_libraries(viewer PRIVATE raylib opengl32 gdi32 winmm)
else()
  find_package(raylib REQUIRED)
  target_link_libraries(viewer PRIVATE raylib)
endif()
