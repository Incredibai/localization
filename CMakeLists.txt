cmake_minimum_required(VERSION 2.8.3)
project(lego_loam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

#set (cloud_msgs_DIR /home/william/catkin_ws/src/LeGO-LOAM)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cloud_msgs
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cloud_msgs
  DEPENDS PCL
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
)

# Localization
add_executable(imageProjection_localization src/localization/imageProjection_localization.cpp)
add_dependencies(imageProjection_localization ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(featureAssociation_localization src/localization/featureAssociation_localization.cpp)
add_dependencies(featureAssociation_localization ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(localizationInMap_localization src/localization/localizationInMap_localization.cpp)
target_link_libraries(localizationInMap_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

add_executable(transformFusion_localization src/localization/transformFusion_localization.cpp)
target_link_libraries(transformFusion_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(publishPCD_localization src/localization/publishPCD_localization.cpp)
target_link_libraries(publishPCD_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(tfListener_localization src/localization/transformListener_localization.cpp)
target_link_libraries(tfListener_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# Mapping
add_executable(imageProjection_mapping src/mapping/imageProjection_mapping.cpp)
add_dependencies(imageProjection_mapping ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(featureAssociation_mapping src/mapping/featureAssociation_mapping.cpp)
add_dependencies(featureAssociation_mapping ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(mapOptimization_mapping src/mapping/mapOptimization_mapping.cpp)
target_link_libraries(mapOptimization_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

add_executable(transformFusion_mapping src/mapping/transformFusion_mapping.cpp)
target_link_libraries(transformFusion_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# KITTI
# Localization
add_executable(imageProjection_localization_K src/KITTI/localization/imageProjection_localization_K.cpp)
add_dependencies(imageProjection_localization_K ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection_localization_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(featureAssociation_localization_K src/KITTI/localization/featureAssociation_localization_K.cpp)
add_dependencies(featureAssociation_localization_K ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation_localization_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(localizationInMap_localization_K src/KITTI/localization/localizationInMap_localization_K.cpp)
target_link_libraries(localizationInMap_localization_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

add_executable(transformFusion_localization_K src/KITTI/localization/transformFusion_localization_K.cpp)
target_link_libraries(transformFusion_localization_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(publishPCD_localization_K src/KITTI/localization/publishPCD_localization_K.cpp)
target_link_libraries(publishPCD_localization_K ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(tfListener_localization_K src/KITTI/localization/transformListener_localization_K.cpp)
target_link_libraries(tfListener_localization_K ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# Mapping
add_executable(imageProjection_mapping_K src/KITTI/mapping/imageProjection_mapping_K.cpp)
add_dependencies(imageProjection_mapping_K ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection_mapping_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(featureAssociation_mapping_K src/KITTI/mapping/featureAssociation_mapping_K.cpp)
add_dependencies(featureAssociation_mapping_K ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation_mapping_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(mapOptimization_mapping_K src/KITTI/mapping/mapOptimization_mapping_K.cpp)
target_link_libraries(mapOptimization_mapping_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

add_executable(transformFusion_mapping_K src/KITTI/mapping/transformFusion_mapping_K.cpp)
target_link_libraries(transformFusion_mapping_K ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

