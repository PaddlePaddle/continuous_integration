project(cpp_inference_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
message("flags" ${CMAKE_CXX_FLAGS})

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()


include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

if (USE_TENSORRT AND WITH_GPU)
      include_directories("${TENSORRT_ROOT}/include")
      link_directories("${TENSORRT_ROOT}/lib")
endif()

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

# add executable
if(NOT ${DEMO_NAME} STREQUAL "all")
  add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
else()
  set(TestFiles "bert_benchmark.cc";
                "clas_benchmark.cc";
                "rcnn_benchmark.cc";
                "ptblm_benchmark.cc";
                "yolo_benchmark.cc";
                "eval_clas.cc";
                "detection_benchmark.cc")
  foreach(testsourcefile ${TestFiles})
    message("====> ${testsourcefile} will be compiled")
    # add executable for all test files
    string(REPLACE ".cc" "" testname ${testsourcefile})
    add_executable(${testname} ${testsourcefile})
  endforeach()
endif()

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
               ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
# Note: libpaddle_fluid.so/a has been changed to libpaddle_inference.so/a after Paddle 2.0.1
if(WITH_STATIC_LIB)
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    glog gflags protobuf xxhash
    ${EXTERNAL_LIB})

if(WITH_GPU)
  if (USE_TENSORRT)
    set(DEPS ${DEPS}
        ${TENSORRT_ROOT}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS}
        ${TENSORRT_ROOT}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(DEPS ${DEPS} ${CUDA_LIB}/libcublas${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# target_link_libraries
if(NOT ${DEMO_NAME} STREQUAL "all")
  target_link_libraries(${DEMO_NAME} ${DEPS})
else()
  foreach(testsourcefile ${TestFiles})
    message("====> ${testsourcefile} will be compiled")
    # add executable for all test files
    string(REPLACE ".cc" "" testname ${testsourcefile})
    target_link_libraries(${testname} ${DEPS})
  endforeach()
endif()
