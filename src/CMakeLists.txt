cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pmconv)

#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#
#add_executable(simple_tensor SimpleTensor.cpp)
#target_link_libraries(simple_tensor "${TORCH_LIBRARIES}")
#set_property(TARGET simple_tensor PROPERTY CXX_STANDARD 20)
#
#add_executable(linear_regression LinearRegression.cpp)
#target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
#set_property(TARGET linear_regression PROPERTY CXX_STANDARD 20)

find_package(Python3 REQUIRED COMPONENTS Development NumPy)

add_executable(example_spline
		example_spline.cpp
		Bspline1D.cpp
		TridiagonalMatrix.cpp
)
target_link_libraries(example_spline PRIVATE Python3::Python Python3::NumPy)
set_property(TARGET example_spline PROPERTY CXX_STANDARD 20)