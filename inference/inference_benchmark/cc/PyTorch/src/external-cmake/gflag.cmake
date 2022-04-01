include(ExternalProject)

set(GFLAG_ROOT          ${CMAKE_BINARY_DIR}/3rdparty/gflag-2.2.2)
set(GFLAG_LIB_DIR       ${GFLAG_ROOT}/lib)
set(GFLAG_INCLUDE_DIR   ${GFLAG_ROOT}/include)

set(GFLAG_URL           https://github.com/gflags/gflags/archive/v2.2.2.zip)
set(GFLAG_CONFIGURE     cd ${GFLAG_ROOT}/src/gflag-2.2.2 && cmake -DCMAKE_INSTALL_PREFIX=${GFLAG_ROOT} -DBUILD_SHARED_LIBS=ON)
set(GFLAG_MAKE          cd ${GFLAG_ROOT}/src/gflag-2.2.2 && make)
set(GFLAG_INSTALL       cd ${GFLAG_ROOT}/src/gflag-2.2.2 && make install)

ExternalProject_Add(gflag-2.2.2
        URL                   ${GFLAG_URL}
        DOWNLOAD_NAME         gflag-2.2.2.zip
        PREFIX                ${GFLAG_ROOT}
        CONFIGURE_COMMAND     ${GFLAG_CONFIGURE}
        BUILD_COMMAND         ${GFLAG_MAKE}
        INSTALL_COMMAND       ${GFLAG_INSTALL}
)
