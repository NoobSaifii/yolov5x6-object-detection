cmake_minimum_required(VERSION 3.10)
project(YOLOv5x6_Object_Detection)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV 4.5 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Executable
add_executable(object_detection main.cpp)

# Link OpenCV libraries
target_link_libraries(object_detection ${OpenCV_LIBS})
