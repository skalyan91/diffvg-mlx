cmake_minimum_required(VERSION 3.15)

project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics" LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CPU-only build: the MLX port has no CUDA code path.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Thrust REQUIRED)
include_directories(${THRUST_INCLUDE_DIR})

if(NOT MSVC)
  # These compile definitions are not meaningful for MSVC
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
endif()

add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
# The vendored thrust refers to libc++'s internal _VSTD macro, which newer libc++ removed.
add_compile_definitions(_VSTD=std)

set(SRCS atomic.h
         color.h
         cdf.h
         cuda_utils.h
         diffvg.h
         edge_query.h
         filter.h
         matrix.h
         parallel.h
         pcg.h
         ptr.h
         sample_boundary.h
         scene.h
         shape.h
         solve.h
         vector.h
         within_distance.h
         winding_number.h
         atomic.cpp
         color.cpp
         diffvg.cpp
         parallel.cpp
         scene.cpp
         shape.cpp)

pybind11_add_module(diffvg MODULE ${SRCS})

set_property(TARGET diffvg PROPERTY CXX_STANDARD 17)
# Still enable assertion in release mode
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

install(TARGETS diffvg LIBRARY DESTINATION .)
