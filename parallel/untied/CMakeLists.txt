cmake_minimum_required(VERSION 3.20)

project(untied VERSION 1.0)

enable_language(CXX)

add_compile_options(-Wall)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(untied untied.cc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(untied PUBLIC OpenMP::OpenMP_CXX)
endif()
