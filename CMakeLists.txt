cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(MOCR)
set(ROOT_DIR        ${CMAKE_CURRENT_SOURCE_DIR})


# --- Sources ---
set(INC_SRC_FILES
    "${ROOT_DIR}/include/mocr/Assert.h"
    "${ROOT_DIR}/include/mocr/Tensor.h"
    "${ROOT_DIR}/include/mocr/Sequential.h"
)

set(SRC_FILES
    "${ROOT_DIR}/src/Layer.h"
    "${ROOT_DIR}/src/Layer.cpp"
    "${ROOT_DIR}/src/Sequential.cpp"
    "${ROOT_DIR}/src/Activations.h"
)

source_group("include" FILES ${INC_SRC_FILES})
source_group("src" FILES ${SRC_FILES})

# --- Library ---
add_library(
    mocr STATIC
    ${INC_SRC_FILES}
    ${SRC_FILES}
)

target_include_directories(
    mocr PUBLIC
    "${ROOT_DIR}/include"
)

target_include_directories(
    mocr PRIVATE
    "${ROOT_DIR}/src"
)

# Sub projects
add_subdirectory(example)

# Resources
set(RES_FILES
	"res/train-images.idx3-ubyte"
	"res/train-labels.idx1-ubyte" 
	"res/t10k-images.idx3-ubyte" 
	"res/t10k-labels.idx1-ubyte" 
)
file(COPY ${RES_FILES} DESTINATION "res")