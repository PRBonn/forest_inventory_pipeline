set(EIGEN_BUILD_DOC OFF CACHE BOOL "Eigen docs")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Eigen tests")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Eigen pkg-config")
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "Eigen Blas module")
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "Eigen Lapack module")

include(FetchContent)
# TODO: add url hash
FetchContent_Declare(eigen URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz SYSTEM
                               EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(eigen)
