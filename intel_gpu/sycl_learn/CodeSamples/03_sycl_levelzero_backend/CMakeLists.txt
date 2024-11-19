if (CMAKE_HOST_WIN32)
	# need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
	cmake_minimum_required(VERSION 3.25)
else()
	# CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
	cmake_minimum_required(VERSION 3.23.5)
endif()

set(PRJ_NAME 03_sycl_levelzero_backend)
project(${PRJ_NAME})

find_package(IntelSYCL REQUIRED)
find_package(OpenCL REQUIRED)

# LevelZero is built from source codes.
#set(LEVELZERO_DIR /mnt/xiping/hw_optimization/intel_gpu/levelzero_learn/CodeSamples/level-zero/build/install)
#set(LZ_INC_DIR ${LEVELZERO_DIR}/include/level_zero)
#set(LZ_LIB_DIR ${LEVELZERO_DIR}/lib)

# oneDNN lib
set(oneDNN_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/oneDNN/build/install/include/)
set(oneDNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/oneDNN/build/install/lib/)

#	${LZ_INC_DIR}

include_directories(
	${Utils}
	${oneDNN_INC_DIR}
	)

file(GLOB SRC 
	"src/*.cpp")

add_executable(${PRJ_NAME}
	${SRC}
)

add_sycl_to_target(TARGET ${PRJ_NAME}  SOURCES ${SRC} )

# ${LZ_LIB_DIR}
target_link_directories(${PRJ_NAME} PUBLIC 
	${oneDNN_LIB_DIR}
	)

#ze_loader
#ze_tracing_layer
#ze_validation_layer

target_link_libraries(${PRJ_NAME} PRIVATE 
	OpenCL::OpenCL 
	dnnl
	)