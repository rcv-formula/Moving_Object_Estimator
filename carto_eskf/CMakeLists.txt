cmake_minimum_required(VERSION 3.8)
project(carto_eskf)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(carto_eskf_node src/carto_eskf_node.cpp)
ament_target_dependencies(carto_eskf_node rclcpp sensor_msgs nav_msgs)
target_link_libraries(carto_eskf_node Eigen3::Eigen)

ament_package()
