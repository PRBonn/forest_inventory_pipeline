add_library(quickshift STATIC)
target_sources(quickshift PRIVATE cluster_remaining.cpp compute_cores.cpp graph_basic.cpp graph.cpp)
target_link_libraries(quickshift PUBLIC Eigen3::Eigen progress)
target_compile_features(quickshift PUBLIC cxx_std_20)
target_compile_options(
  quickshift
  PRIVATE # GNU
          $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>
          $<$<COMPILE_LANGUAGE:CXX>:-Werror>
          $<$<COMPILE_LANGUAGE:CXX>:-Wall>
          $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
          $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
          $<$<COMPILE_LANGUAGE:CXX>:-Wcast-align>
          $<$<COMPILE_LANGUAGE:CXX>:-Wcast-qual>
          # $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
          $<$<COMPILE_LANGUAGE:CXX>:-Wdisabled-optimization>
          $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>)

target_include_directories(quickshift PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
