cmake_minimum_required(VERSION 3.8)
project(obstacle_context)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(tf2 REQUIRED)

find_package(obstacle_context_msgs REQUIRED)

add_executable(obstacle_cluster_context_node
  src/obstacle_cluster_context_node.cpp
)

ament_target_dependencies(obstacle_cluster_context_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  obstacle_context_msgs
)

install(TARGETS
  obstacle_cluster_context_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()