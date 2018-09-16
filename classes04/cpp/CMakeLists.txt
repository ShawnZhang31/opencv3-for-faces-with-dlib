cmake_minimum_required(VERSION 2.8.12)

PROJECT(classes04)



#############  基本库配置  ############
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})
include(../../common/dlib/dlib/cmake)

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()


add_example(alphaBlend)
add_example(warpTriangle)
add_example(delaunay)
add_example(delaunayAnimation)
add_example(faceAveraging)
add_example(faceMorph)
add_example(blinkDetect)