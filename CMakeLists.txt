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

add_executable(solve tools/solve.cpp)
target_link_libraries(solve PRIVATE gdcore)

# Viewer GUI deps (Dear ImGui + rlImGui backend + nlohmann/json), fetched at
# configure time into build/_deps. Pins: rlImGui tag Raylib_5_5 (main targets
# raylib 6.0 and calls GetDisplayScale, absent in 5.5) which needs ImGui >= 1.92
# (dynamic-texture backend API). Neither ships usable CMake; their sources are
# compiled straight into the viewer. Set VIEWER_VENDORED_DEPS=<dir> holding
# imgui/, rlImGui/, json/ checkouts to build offline.
include(FetchContent)
set(VIEWER_VENDORED_DEPS "" CACHE PATH "offline checkout dir for imgui/rlImGui/json")
if(VIEWER_VENDORED_DEPS)
  set(imgui_SOURCE_DIR ${VIEWER_VENDORED_DEPS}/imgui)
  set(rlimgui_SOURCE_DIR ${VIEWER_VENDORED_DEPS}/rlImGui)
  set(json_SOURCE_DIR ${VIEWER_VENDORED_DEPS}/json)
else()
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG dac07199cfd761113d966eb8ad739254e10df2fe  # v1.92.7
    GIT_SHALLOW OFF)
  FetchContent_Declare(rlimgui
    GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
    GIT_TAG 118221c8a532501a3c2c236b62950877cbefc2b4  # tag Raylib_5_5
    GIT_SHALLOW OFF)
  FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW ON)
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(imgui rlimgui json)
endif()

set(VIEWER_THIRD_PARTY
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${rlimgui_SOURCE_DIR}/rlImGui.cpp
)
set_source_files_properties(${VIEWER_THIRD_PARTY} PROPERTIES COMPILE_OPTIONS "-w")

add_executable(viewer
  src/viewer/main.cpp
  src/viewer/atlas.cpp
  src/viewer/world.cpp
  src/viewer/modes.cpp
  src/viewer/modes_rl.cpp
  src/viewer/gui.cpp
  src/viewer/jobs.cpp
  src/viewer/paths.cpp
  src/viewer/prefs.cpp
  src/viewer/runs.cpp
  ${VIEWER_THIRD_PARTY}
)
target_include_directories(viewer PRIVATE
  ${imgui_SOURCE_DIR} ${rlimgui_SOURCE_DIR} ${json_SOURCE_DIR}/include)
target_compile_definitions(viewer PRIVATE NO_FONT_AWESOME IMGUI_DISABLE_DEMO_WINDOWS
  IMGUI_DISABLE_WIN32_DEFAULT_IME_FUNCTIONS)
target_link_libraries(viewer PRIVATE gdcore)
if(MINGW)
  target_link_libraries(viewer PRIVATE raylib opengl32 gdi32 winmm shell32 ole32)
else()
  find_package(raylib REQUIRED)
  target_link_libraries(viewer PRIVATE raylib)
endif()
