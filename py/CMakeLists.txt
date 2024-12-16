cmake_minimum_required(VERSION 3.16)
project(implicit2d)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include_directories(../cpp/lib/include/)

find_library(IMPL impl PATHS ../bin/ REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(implicit2d src/binder.cxx)

target_link_libraries(${PROJECT_NAME} PRIVATE ${IMPL}) # The PRIVATE keyword is important

install(TARGETS implicit2d DESTINATION .)
