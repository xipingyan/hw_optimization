cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

project(SyclSamples LANGUAGES CXX)

set(Utils  ${CMAKE_SOURCE_DIR}/utils)

add_subdirectory(01_HelloSycl)
add_subdirectory(02_sycl_ocl_interoperate)
add_subdirectory(03_sycl_levelzero_backend)
add_subdirectory(04_USM_TerseSyntax)
add_subdirectory(05_Sycl_Concept)
add_subdirectory(06_Sycl_Extension_SIMD)
add_subdirectory(07_ExecuteOrder)

# Test my implementation kernel.
add_subdirectory(kernel_01_MatMul)