cmake_minimum_required(VERSION 3.0)
project(inference_test CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       OFF)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

include_directories("${PADDLE_LIB}/")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}zlib/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")
include_directories("${PADDLE_LIB}/paddle/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}zlib/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -std=c++11")
message("flags" ${CMAKE_CXX_FLAGS})

if (USE_TENSORRT AND WITH_GPU)
  include_directories("${TENSORRT_INCLUDE_DIR}")
  link_directories("${TENSORRT_LIB_DIR}")
endif()

set(NGRAPH_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}ngraph")
if(EXISTS ${NGRAPH_PATH})
    include(GNUInstallDirs)
    include_directories("${NGRAPH_PATH}/include")
    link_directories("${NGRAPH_PATH}/${CMAKE_INSTALL_LIBDIR}")
    set(NGRAPH_LIB ${NGRAPH_PATH}/${CMAKE_INSTALL_LIBDIR}/libngraph${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if(WITH_MKL)
  set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
  include_directories("${MATH_LIB_PATH}/include")
  set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  set(MATH_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

set(EXTERNAL_LIB "-lrt -ldl -lpthread -lprotobuf")
set(DEPS ${DEPS}
  ${MATH_LIB} ${MKLDNN_LIB} ${NGRAPH_LIB}
  glog gflags protobuf z xxhash
  ${EXTERNAL_LIB})

if(WITH_GPU)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} $ENV{CUDA_LIB})
endif()

include(ExternalProject)
include(external-cmake/gtest-cpp.cmake)
add_executable("test_resnet50" test_resnet50.cc)
target_link_libraries("test_resnet50" ${DEPS})
add_dependencies("test_resnet50" thirdparty_gtest)

# link gtest
set(gtest_lib_dir "${CMAKE_CURRENT_BINARY_DIR}/install/gtest/lib64")
set(gtest_lib_dir "${CMAKE_CURRENT_BINARY_DIR}/install/gtest/lib")
set(gtest_src_dir "${CMAKE_CURRENT_BINARY_DIR}/install/gtest/include")
target_include_directories("test_resnet50" PRIVATE  ${gtest_src_dir})
target_include_directories("test_resnet50" PRIVATE ${gtest_lib_dir})
target_link_libraries("test_resnet50" ${gtest_lib_dir}/libgtest.a)
