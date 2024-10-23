add_executable(00_HelloLevelZero
	src/main.cpp
)

target_include_directories(00_HelloLevelZero PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})