cmake_minimum_required(VERSION 2.8.12)

PROJECT(classes05)



#############  基本库配置  ############
find_package( OpenCV REQUIRED )
find_package(dlib REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TURE)

include_directories( ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "dlib library status:")
message(STATUS "    config: ${dlib_DIR}")
message(STATUS "    version: ${dlib_VERSION}")

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()

add_example(clone)
add_example(faceswapImage)
