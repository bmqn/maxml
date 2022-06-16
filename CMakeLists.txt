cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(maxml)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
add_compile_options(
	$<$<CONFIG:RELEASE>:/Ox>
	$<$<CONFIG:RELEASE>:/GL>
)
add_link_options(
	$<$<CONFIG:RELEASE>:/LTCG>
)
else()
add_compile_options(
	$<$<CONFIG:RELEASE>:-Ofast>
)
add_link_options(
	$<$<CONFIG:RELEASE>:-flto>
)
endif()


# --- Sources ---
set(INC_FILES
	"${ROOT_DIR}/include/maxml/MmlTensor.h"
	"${ROOT_DIR}/include/maxml/MmlSequential.h"
)

set(SRC_FILES
	"${ROOT_DIR}/src/MmlConfig.h"
	"${ROOT_DIR}/src/MmlLog.h"
	"${ROOT_DIR}/src/MmlLayer.h"
	"${ROOT_DIR}/src/MmlLayer.cpp"
	"${ROOT_DIR}/src/MmlSequential.cpp"
	"${ROOT_DIR}/src/MmlTensor.cpp"
	"${ROOT_DIR}/src/MmlActivations.h"
)

source_group("Include" FILES ${INC_FILES})
source_group("Source"  FILES ${SRC_FILES})


# --- Library ---
add_library(
	maxml STATIC
	${INC_FILES}
	${SRC_FILES}
)

target_include_directories(
	maxml
	PUBLIC  "${ROOT_DIR}/include"
	PRIVATE "${ROOT_DIR}/src"
)

# --- Example ---
add_executable(
	example
	"${ROOT_DIR}/example/Main.cpp"
	"${ROOT_DIR}/example/MnistLoader.h"
)

target_link_libraries(
	example
	PUBLIC maxml
)