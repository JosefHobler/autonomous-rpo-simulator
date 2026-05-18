cmake_minimum_required(VERSION 3.18)
project(rpo_ekf_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED CONFIG)

add_executable(ekf_smoke tests/test_ekf_core.cpp)
target_include_directories(ekf_smoke PRIVATE include)
target_link_libraries(ekf_smoke PRIVATE Eigen3::Eigen)

find_package(pybind11 CONFIG QUIET)
if (pybind11_FOUND)
    pybind11_add_module(_ekf_cpp src/pybindings.cpp)
    target_include_directories(_ekf_cpp PRIVATE include)
    target_link_libraries(_ekf_cpp PRIVATE Eigen3::Eigen)
    set_target_properties(_ekf_cpp PROPERTIES OUTPUT_NAME "_ekf_cpp")
else()
    message(STATUS "pybind11 not found - building C++ smoke test only.")
endif()
