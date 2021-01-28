cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
find_package(TensorflowCC REQUIRED)

include(external-cmake/gflag.cmake)
include(external-cmake/glog.cmake)

include_directories(${CMAKE_BINARY_DIR}/3rdparty/glog/include)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/gflag-2.2.2/include)

link_directories(${CMAKE_BINARY_DIR}/3rdparty/glog/lib)
link_directories(${CMAKE_BINARY_DIR}/3rdparty/gflag-2.2.2/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(EXTERNAL_LIB "-lrt -ldl -lprotobuf")

add_executable(clas_benchmark clas_benchmark.cc)
add_dependencies(clas_benchmark glog gflag-2.2.2)
target_link_libraries(clas_benchmark TensorflowCC::TensorflowCC ${EXTERNAL_LIB})


# link cuda if it is available
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(clas_benchmark ${CUDA_LIBRARIES})
endif()

# link gflags
target_link_libraries(clas_benchmark gflags)
target_link_libraries(clas_benchmark libglog.so)

# find_package (glog 0.4.0 REQUIRED)
# target_link_libraries(clas_benchmark glog::glog)
