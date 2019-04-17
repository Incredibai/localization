cmake_minimum_required(VERSION 2.8.3)
project(kitti_parser)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
  pcl_conversions)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  #${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs tf pcl_conversions
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS include
)

add_library(matrix
  include/matrix.h src/matrix.cpp )

target_link_libraries(matrix ${catkin_LIBRARIES})
add_dependencies(matrix ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Compile as C++14, supported in ROS Kinetic and newer
# set_property(TARGET invz_player PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions( -march=native )

add_executable(kitti_parser src/kitti_parser_node.cpp)
target_link_libraries(kitti_parser ${catkin_LIBRARIES} ${PCL_LIBRARIES} matrix)

add_executable(kitti_parser_localization src/kitti_parser_localization.cpp)
target_link_libraries(kitti_parser_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} matrix)

