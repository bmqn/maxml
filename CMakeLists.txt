cmake_minimum_required(VERSION 3.1.0)

set(MML_ROOT_DIR		${CMAKE_CURRENT_SOURCE_DIR})
set(MML_INC_DIR			${MML_ROOT_DIR}/include)
set(MML_SRC_DIR			${MML_ROOT_DIR}/src)
set(MML_RES_DIR			${MML_ROOT_DIR}/res)

project(maxml)

#--------------------------------------------------------------------------------------------------
#	Configuration
#--------------------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

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

#--------------------------------------------------------------------------------------------------
#	Sources
#--------------------------------------------------------------------------------------------------
set(MML_HSP
	"${MML_INC_DIR}/maxml/MmlTensor.h"
	"${MML_INC_DIR}/maxml/MmlSequential.h"
)

set(MML_SRC
	"${MML_SRC_DIR}/MmlPrefix.pch"

	"${MML_SRC_DIR}/MmlConfig.h"
	"${MML_SRC_DIR}/MmlLog.h"
	"${MML_SRC_DIR}/MmlUtils.h"
	"${MML_SRC_DIR}/MmlLayer.h"
	"${MML_SRC_DIR}/MmlLayer.cpp"
	"${MML_INC_DIR}/maxml/MmlSequential.h"
	"${MML_SRC_DIR}/MmlSequential.cpp"
	"${MML_INC_DIR}/maxml/MmlTensor.h"
	"${MML_SRC_DIR}/MmlTensor.cpp"
)

source_group("Include" FILES ${MML_HSP})
source_group("Source"  FILES ${MML_SRC})


#--------------------------------------------------------------------------------------------------
#	Build
#--------------------------------------------------------------------------------------------------
add_library(
	maxml STATIC
	${MML_HSP}
	${MML_SRC}
)

target_include_directories(
	maxml
	PUBLIC  ${MML_INC_DIR}
	PRIVATE ${MML_SRC_DIR}
)

target_precompile_headers(
	maxml
	PRIVATE "${MML_SRC_DIR}/MmlPrefix.pch"
)

add_executable(
	example
	"${MML_ROOT_DIR}/example/Main.cpp"
)

target_link_libraries(
	example
	PUBLIC maxml
)

set_target_properties(example PROPERTIES
	VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:example>
)

#--------------------------------------------------------------------------------------------------
#	Resources
#--------------------------------------------------------------------------------------------------
add_custom_target(maxml_CopyResources ALL
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${MML_RES_DIR} $<TARGET_FILE_DIR:maxml>
)