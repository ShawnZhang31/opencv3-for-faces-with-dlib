cmake_minimum_required(VERSION 2.8.12)

PROJECT(classes01)


######################## 如果需要编辑一下 ####################
#SET(OpenCV_DIR 如需opencv的安装位置)


################### opencv安装位置示例  #####################

### MACOS : /usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/

### UBUNTU : /usr/local/share/OpenCV/

### WINDOWS : C:\Users\yourname\Documents\opencv-3.3.1\build\install

##############################################################




###################   ***不要修改***  #####################

#############  Common Instructions for all Users  ############
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} )
ENDMACRO()

add_example(readWriteDisplay)
add_example(cropAndResize)
add_example(datatypeConversion)
add_example(imageDraw)
