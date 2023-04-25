project(ncnnhair)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc photo highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20221128-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnnhair SHARED nanodetncnn.cpp nanodet.cpp ndkcamera.cpp )

target_link_libraries(ncnnhair ncnn ${OpenCV_LIBS} camera2ndk mediandk)
