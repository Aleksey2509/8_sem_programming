cmake_minimum_required(VERSION 3.20)

project(send VERSION 1.0)

enable_language(CXX)

add_compile_options(-Wall)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(send send.cc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(send PUBLIC OpenMP::OpenMP_CXX)
endif()
