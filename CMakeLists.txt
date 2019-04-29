cmake_minimum_required(VERSION 2.8.3)
project(face_sfm)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  tf
  rosbag
  pcl_ros
  message_filters
  sensor_msgs
)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_executable(${PROJECT_NAME}
	       src/face_sfm.cpp
	       src/camera.cpp
	       src/ceres/util.cpp
	       src/ceres/local_parameterization_se3.cpp
	       src/ceres/factor.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 


