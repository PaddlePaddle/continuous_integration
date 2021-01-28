include(ExternalProject)

set(GLOG_ROOT          ${CMAKE_BINARY_DIR}/3rdparty/glog)
set(GLOG_LIB_DIR       ${GLOG_ROOT}/lib)
set(GLOG_INCLUDE_DIR   ${GLOG_ROOT}/include)

set(GLOG_URL           https://github.com/google/glog/archive/d516278b1cd33cd148e8989aec488b6049a4ca0b.zip)
set(GLOG_CONFIGURE     cd ${GLOG_ROOT}/src/glog && cmake -DCMAKE_INSTALL_PREFIX=${GLOG_ROOT} -DBUILD_SHARED_LIBS=ON)
set(GLOG_MAKE          cd ${GLOG_ROOT}/src/glog && make)
set(GLOG_INSTALL       cd ${GLOG_ROOT}/src/glog && make install)

ExternalProject_Add(glog
        URL                   ${GLOG_URL}
        DOWNLOAD_NAME         glog-dev
        PREFIX                ${GLOG_ROOT}
        CONFIGURE_COMMAND     ${GLOG_CONFIGURE}
        BUILD_COMMAND         ${GLOG_MAKE}
        INSTALL_COMMAND       ${GLOG_INSTALL}
)
