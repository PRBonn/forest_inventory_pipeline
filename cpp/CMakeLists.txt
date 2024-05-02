include(${CMAKE_CURRENT_LIST_DIR}/cmake/StandardProjectOptions.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/3rdparty/find_dependencies.cmake)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/quickshift)
if(SKBUILD)
  # gaurd prevents in cpp only build
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/quickshift_pybind)
endif()
