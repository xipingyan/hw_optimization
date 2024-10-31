cmake_minimum_required(VERSION 3.0)

set(PRJ_NAME 01_HelloOpenCL)
project(${PRJ_NAME} LANGUAGES C)

find_package(OpenCL REQUIRED)

file(GLOB SRC 
	"src/*.cpp")

#MESSAGE(STATUS "Utils=${Utils}")

add_executable(${PRJ_NAME} ${SRC})
target_link_libraries(${PRJ_NAME} PRIVATE OpenCL::OpenCL)