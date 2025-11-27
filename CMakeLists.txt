cmake_minimum_required(VERSION 3.10)
project(golf_ball_launch_calculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

file(GLOB SOURCES src/*.cpp)

add_executable(main ${SOURCES})

target_link_libraries(main PRIVATE ${OpenCV_LIBS})
