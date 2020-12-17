cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp_inference_torch CXX C)

if(NOT DEFINED TORCH_LIB)
  message(FATAL_ERROR "please set TORCH_LIB with -DTORCH_LIB=/path/torch/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

find_package(Torch REQUIRED
             PATHS ${TORCH_LIB} # find torch lib here
             NO_DEFAULT_PATH)
message("TORCH_LIBRARIES is ${TORCH_LIB}")

include(external-cmake/gflag.cmake)
include(external-cmake/glog.cmake)

include_directories(${CMAKE_BINARY_DIR}/3rdparty/glog/include)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/gflag-2.2.2/include)

link_directories(${CMAKE_BINARY_DIR}/3rdparty/glog/lib)
link_directories(${CMAKE_BINARY_DIR}/3rdparty/gflag-2.2.2/lib)

# add executable
if(NOT ${DEMO_NAME} STREQUAL "all")
  add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
  add_dependencies(${DEMO_NAME} glog gflag-2.2.2)
  target_link_libraries(${DEMO_NAME} "${TORCH_LIBRARIES}")
  target_link_libraries(${DEMO_NAME} gflags)
  target_link_libraries(${DEMO_NAME} libglog.so)
  set_property(TARGET ${DEMO_NAME} PROPERTY CXX_STANDARD 14)
else()
  set(TestFiles "clas_benchmark.cc")
  foreach(testsourcefile ${TestFiles})
    message("====> ${testsourcefile} will be compiled")
    # add executable for all test files
    string(REPLACE ".cc" "" testname ${testsourcefile})
    add_executable(${testname} ${testsourcefile})
    add_dependencies(${testname} glog gflag-2.2.2)
    target_link_libraries(${testname} "${TORCH_LIBRARIES}")
    target_link_libraries(${testname} gflags)
    target_link_libraries(${testname} libglog.so)
    set_property(TARGET ${testname} PROPERTY CXX_STANDARD 14)
  endforeach()
endif()
