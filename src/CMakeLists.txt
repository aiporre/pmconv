cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pmconv)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(simple_tensor SimpleTensor.cpp)
target_link_libraries(simple_tensor "${TORCH_LIBRARIES}")
set_property(TARGET simple_tensor PROPERTY CXX_STANDARD 20)

add_executable(linear_regression LinearRegression.cpp)
target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
set_property(TARGET linear_regression PROPERTY CXX_STANDARD 20)

# --------------------------------------------------------------------------
# pmconv_ext — Python extension (.so) built with pybind11 + LibTorch
#
# Preferred build method: use setup.py (pip install .) which invokes
# torch.utils.cpp_extension and handles all include/link paths automatically.
#
# This CMake target is provided as an alternative for toolchains that prefer
# a pure-CMake workflow.  You must supply the paths to Python, pybind11 and
# the LibTorch CMake config when invoking CMake, e.g.
#
#   cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" ..
# --------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(pmconv_ext
        spline_conv.cpp
        pmconv_bindings.cpp
    )
    target_link_libraries(pmconv_ext PRIVATE "${TORCH_LIBRARIES}")
    set_property(TARGET pmconv_ext PROPERTY CXX_STANDARD 17)
    message(STATUS "pmconv_ext target configured via CMake + pybind11")
else()
    message(STATUS "pybind11 not found via CMake — use 'pip install .' instead")
endif()