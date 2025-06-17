cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

project(CMSamples LANGUAGES CXX)

set(LevelZeroHeader ${CMAKE_SOURCE_DIR}/level-zero/build/install/include)
set(LevelZeroLib ${CMAKE_SOURCE_DIR}/level-zero/build/install/lib)
set(Utils  ${CMAKE_SOURCE_DIR}/utils)

add_subdirectory(01_HelloCM)