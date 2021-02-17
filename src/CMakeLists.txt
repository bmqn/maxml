cmake_minimum_required(VERSION 3.1.0)

file(GLOB_RECURSE SRC_FILES "*.cpp")
file(GLOB_RECURSE HDR_FILES "*.h")

add_library(
    maxocr STATIC
    ${SRC_FILES})

target_include_directories(
    maxocr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
