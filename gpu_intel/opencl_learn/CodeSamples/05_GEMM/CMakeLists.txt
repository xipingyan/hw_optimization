cmake_minimum_required(VERSION 3.0)

set(PRJ_NAME 05_GEMM)
project(${PRJ_NAME} LANGUAGES C)

find_package(OpenCL REQUIRED)

include_directories(
	${Utils}
	)

file(GLOB SRC 
	"src/*.cpp")

add_executable(${PRJ_NAME} ${SRC})
target_link_libraries(${PRJ_NAME} PRIVATE OpenCL::OpenCL)