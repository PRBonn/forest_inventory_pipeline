set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_quickshift_pybind MODULE quickshift_pybind.cpp)
target_link_libraries(_quickshift_pybind PRIVATE quickshift)

install(TARGETS _quickshift_pybind DESTINATION forest_inventory_pipeline/cluster/quickshiftpp)
