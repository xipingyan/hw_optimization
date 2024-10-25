cmake_minimum_required(VERSION 3.27)

set(PRJ_NAME 00_HelloLevelZero)
project(${PRJ_NAME})

include_directories(
	${Utils}
	${LevelZeroHeader}
	)

file(GLOB SRC 
	"${LevelZeroHeader}/*.h"
	"${LevelZeroHeader}/*.hpp"
	"src/*.cpp")

MESSAGE(STATUS "LevelZeroHeader=${LevelZeroHeader}")
MESSAGE(STATUS "Utils=${Utils}")

add_executable(${PRJ_NAME}
	${SRC}
)

target_link_directories(${PRJ_NAME} PUBLIC 
	${LevelZeroLib})
target_link_libraries(${PRJ_NAME} PRIVATE 
	ze_loader
	ze_tracing_layer
	ze_validation_layer
	)