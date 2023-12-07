# Specify the minimum required CMake version and the project name
cmake_minimum_required(VERSION 3.17)
project(pose-estimation2)

# Set C++ standard to 14
set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
if(WIN32)
    # Set OpenCV directory for Windows
    set(OpenCV_DIR "E:/Graduation_project/TASKs/1.MOVENET/Git/tflite/opencv/build")
endif()

# Find and include OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)

# Add the main.cpp file to the executable
add_executable(${PROJECT_NAME}  main/main.cpp)

# For Windows platform
if(WIN32)
    # Find and link TensorFlow Lite and FlatBuffers libraries
    find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)
    find_library(FLATBUFFERS flatbuffers  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TFLITE} ${FLATBUFFERS} ${OpenCV_LIBS})

    # Copy required DLLs to the binary directory after building
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/tensorflowlite.dll ${CMAKE_BINARY_DIR}/Release/
    )
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenCV_DIR}/x64/vc16/bin/opencv_world480.dll ${CMAKE_BINARY_DIR}/Release/
    )
endif()

# For Linux platform
if(UNIX AND NOT APPLE)
    # Add TensorFlow Lite library for Linux
    ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
    set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
    target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})
endif()

# For macOS platform
if(APPLE)
    # Add TensorFlow Lite library for macOS
    ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
    set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/mac/libtensorflowlite.dylib)
    target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})
endif()
