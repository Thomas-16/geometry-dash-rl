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
  src/core/geometry.cpp
)
target_include_directories(gdcore PUBLIC src)
target_link_libraries(gdcore PUBLIC ZLIB::ZLIB)

# Training-env DLL: C ABI over gdcore for the Python trainer (ctypes).
# MODULE (runtime-loaded only; nothing links against it): this machine's ld
# exit-116 bug fires on -shared combined with either the global -static or
# --out-implib, so the target clears the inherited -static and statically
# links the runtimes piecewise instead (whole-archive winpthread beats the
# driver's trailing shared -lpthread). Result imports only KERNEL32 + UCRT
# api-sets, so any Python can load it without ucrt64 DLLs on PATH.
add_library(gdenv MODULE src/env/gdenv_c.cpp)
target_link_libraries(gdenv PRIVATE gdcore)
target_compile_definitions(gdenv PRIVATE GDENV_BUILD)
set_target_properties(gdenv PROPERTIES PREFIX "")
if(MINGW)
  set_target_properties(gdenv PROPERTIES LINK_OPTIONS "")
  target_link_options(gdenv PRIVATE -static-libgcc -static-libstdc++
    "LINKER:-Bstatic,--whole-archive" -lwinpthread
    "LINKER:--no-whole-archive")
endif()

add_executable(env_check tools/env_check.cpp)
target_link_libraries(env_check PRIVATE gdcore)

add_executable(dump tools/dump.cpp)
target_link_libraries(dump PRIVATE gdcore)

add_executable(trace_check tools/trace_check.cpp)
target_link_libraries(trace_check PRIVATE gdcore)

add_executable(viewer
  src/viewer/main.cpp
  src/viewer/atlas.cpp
  src/viewer/world.cpp
  src/viewer/modes.cpp
  src/viewer/modes_rl.cpp
)
target_link_libraries(viewer PRIVATE gdcore)
if(MINGW)
  target_link_libraries(viewer PRIVATE raylib opengl32 gdi32 winmm)
else()
  find_package(raylib REQUIRED)
  target_link_libraries(viewer PRIVATE raylib)
endif()
