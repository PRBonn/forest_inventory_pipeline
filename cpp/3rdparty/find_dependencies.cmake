# Modified from kiss-icp. following license is kiss-icp license
# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
function(find_dep PACKAGE_NAME TARGET_NAME FETCH_CMAKE_FILE)
  if(NOT TARGET ${TARGET_NAME})
    message(STATUS "${PACKAGE_NAME}'s target: ${TARGET_NAME} not found.")
    string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UP)
    set(USE_FROM_SYSTEM_OPTION "USE_SYSTEM_${PACKAGE_NAME_UP}")
    if(${${USE_FROM_SYSTEM_OPTION}})
      message(STATUS "Searching for ${PACKAGE_NAME} in system. ")
      find_package(${PACKAGE_NAME} QUIET)
    endif()
    if(NOT TARGET ${TARGET_NAME})
      message(STATUS "Fetching it using FetchContent (${FETCH_CMAKE_FILE})")
      include(${FETCH_CMAKE_FILE})
    endif()
  endif()
endfunction()

option(USE_SYSTEM_EIGEN3 "Use system Eigen" ON)
find_dep("Eigen3" "Eigen3::Eigen" "${CMAKE_CURRENT_LIST_DIR}/eigen/eigen.cmake")
find_dep("progress" "progress" "${CMAKE_CURRENT_LIST_DIR}/progress/progress.cmake")
