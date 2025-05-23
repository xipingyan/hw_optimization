if (CMAKE_HOST_WIN32)
	# need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
	cmake_minimum_required(VERSION 3.25)
else()
	# CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
	cmake_minimum_required(VERSION 3.23.5)
endif()

set(PRJ_NAME 02_sycl_ocl_interoperate)
project(${PRJ_NAME})

MESSAGE(STATUS "== Start: ${PRJ_NAME}")

find_package(IntelSYCL REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(
	${Utils}
	)

file(GLOB SRC 
	"src/*.cpp")

add_executable(${PRJ_NAME}
	${SRC}
)

add_sycl_to_target(TARGET ${PRJ_NAME}  SOURCES ${SRC} )
target_link_libraries(${PRJ_NAME} PRIVATE OpenCL::OpenCL
ze_loader
ze_tracing_layer
ze_validation_layer)