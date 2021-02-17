cmake_minimum_required(VERSION 3.1.0)

add_executable(
    example
    Main.cpp)

target_link_libraries(
    example PUBLIC
    maxocr)