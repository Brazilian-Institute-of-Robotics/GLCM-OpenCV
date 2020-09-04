CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(GLCM)
set(CMAKE_CXX_STANDARD 11)
# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
# SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

FIND_PACKAGE(OpenCV REQUIRED)
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/lib)
FIND_PACKAGE(OpenCL REQUIRED 2.0)


set(name glcm)
set(OCL_NAME opencl_kernels_${name})

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp"  # don't add .hpp file here to optimize build process
    COMMAND ${CMAKE_COMMAND} "-DMODULE_NAME=${name}" "-DCL_DIR=${CMAKE_CURRENT_LIST_DIR}/opencl" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cl2cpp.cmake"
    DEPENDS ./cmake/cl2cpp.cmake
    COMMENT "Processing OpenCL kernels (${name})"
)

set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp"
    PROPERTIES GENERATED TRUE
)

list(APPEND lib_srcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ./
)
INCLUDE_DIRECTORIES(./src)

SET(LIB_GLCM_SOURCE
	./src/glcm.h
	./src/glcm.cpp)
ADD_LIBRARY(glcm SHARED ${LIB_GLCM_SOURCE})
TARGET_LINK_LIBRARIES(glcm
	${OpenCV_LIBS})

SET(LIB_GLCM glcm)
ADD_EXECUTABLE(GLCM_test ./src/main.cpp)
TARGET_LINK_LIBRARIES(GLCM_test
    ${LIB_GLCM})
if(OpenCL_FOUND)
    add_definitions(-DHAVE_OPENCL)
endif()
ADD_LIBRARY(glcm_dispatch ./src/glcm_dispatch.cpp ${lib_srcs})
TARGET_LINK_LIBRARIES(glcm_dispatch
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES})
    
install(TARGETS glcm
    LIBRARY DESTINATION lib)

install(DIRECTORY src/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*~" EXCLUDE
    )

configure_file(cmake/GLCM-config.cmake.in GLCMConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GLCMConfig.cmake
    DESTINATION share/GLCM/cmake)


if (BIR_ENABLE_TESTING)
    enable_testing()
    find_package(bir_cmake_modules REQUIRED)
    find_package(benchmark REQUIRED)

    bir_add_test_gtest(test_opencl_execution test/test_opencl_execution.cpp)
    target_link_libraries(test_opencl_execution glcm_dispatch)

    ADD_EXECUTABLE(benchmark_opencl_execution test/benchmark_opencl_execution.cpp)
    target_link_libraries(benchmark_opencl_execution glcm_dispatch benchmark::benchmark)

endif()
