cmake_minimum_required(VERSION 3.1.0)

file(GLOB_RECURSE RES_FILES "res/*.idx3-ubyte" "res/*.idx1-ubyte")
file(COPY ${RES_FILES} DESTINATION "res")

project(MaxOCR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)
add_subdirectory(example)