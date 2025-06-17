if (CMAKE_HOST_WIN32)
	# need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
	cmake_minimum_required(VERSION 3.25)
else()
	# CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
	cmake_minimum_required(VERSION 3.23.5)
endif()

set(PRJ_NAME 01_HelloCM)
project(${PRJ_NAME})

MESSAGE(STATUS "== Start: ${PRJ_NAME}")

if (NOT DEFINED Utils)
	set(Utils ${CMAKE_SOURCE_DIR}/../utils)
endif()
MESSAGE(STATUS "Utils = ${Utils}")

include_directories(
	${Utils}
	${LevelZeroHeader}
	)

file(GLOB SRC 
	"${LevelZeroHeader}/*.h"
	"${LevelZeroHeader}/*.hpp"
	"src/*.cpp")

add_executable(${PRJ_NAME}
	${SRC}
)

target_link_libraries(${PRJ_NAME} PRIVATE 
	ze_loader
	ze_tracing_layer
	ze_validation_layer
	)