cmake_minimum_required(VERSION 3.5)
project(object_detector)

# --- (선택) MPI 고정: 필요 없으면 제거 가능 ---
set(MPI_C_COMPILER "/usr/bin/mpicc" CACHE FILEPATH "MPI C compiler path")
set(MPI_CXX_COMPILER "/usr/bin/mpicxx" CACHE FILEPATH "MPI C++ compiler path")
set(MPI_HOME "/usr/lib/aarch64-linux-gnu/openmpi" CACHE PATH "MPI home path")
set(CMAKE_IGNORE_PREFIX_PATH "/opt/hpcx" CACHE PATH "Ignore broken HPCX path")
find_package(MPI REQUIRED COMPONENTS C CXX)
include_directories(${MPI_C_INCLUDE_PATH})

# --- ament & ROS deps ---
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)                 # ★ DEPENDENCIES에 쓰므로 필수
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)          # ★ 사용함
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(PCL REQUIRED COMPONENTS common io filters visualization)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)

# --- 메시지 인터페이스 생성 ---
set(MSG_FILES
  "msg/MarkerArrayStamped.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  DEPENDENCIES std_msgs visualization_msgs
)

# --- include 경로 ---
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 헤더 전용 타겟(공유 include용)
add_library(object_detector_headers INTERFACE)
target_include_directories(object_detector_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# === 실행 파일들 ===

# 1) scan_processor_node
add_executable(scan_processor_node src/scan_processor_node.cpp)
ament_target_dependencies(scan_processor_node
  rclcpp sensor_msgs geometry_msgs nav_msgs
  visualization_msgs pcl_conversions
  tf2_ros tf2_geometry_msgs
  Eigen3
)
target_link_libraries(scan_processor_node ${PCL_LIBRARIES})
rosidl_target_interfaces(scan_processor_node ${PROJECT_NAME} "rosidl_typesupport_cpp")
add_dependencies(scan_processor_node ${PROJECT_NAME}__rosidl_typesupport_cpp)  # ★ 빌드 순서 보강

# 2) obstacle_detector_node
add_executable(obstacle_detector_node src/obstacle_detector_node.cpp)
ament_target_dependencies(obstacle_detector_node
  rclcpp geometry_msgs nav_msgs visualization_msgs
  pcl_conversions tf2_ros tf2_geometry_msgs
)
target_link_libraries(obstacle_detector_node object_detector_headers ${PCL_LIBRARIES})
rosidl_target_interfaces(obstacle_detector_node ${PROJECT_NAME} "rosidl_typesupport_cpp")
add_dependencies(obstacle_detector_node ${PROJECT_NAME}__rosidl_typesupport_cpp)

# 3) visualization_node
add_executable(visualization_node src/visualization_node.cpp)
ament_target_dependencies(visualization_node
  rclcpp nav_msgs visualization_msgs
)

# 4) delay_monitor_node
add_executable(delay_monitor_node src/delay_monitor_node.cpp)
ament_target_dependencies(delay_monitor_node
  rclcpp sensor_msgs nav_msgs message_filters geometry_msgs
)

# 5) icp_comparator_node
add_executable(icp_comparator_node src/icp_comparator_node.cpp)
ament_target_dependencies(icp_comparator_node
  rclcpp sensor_msgs nav_msgs std_msgs geometry_msgs
  message_filters pcl_conversions
)
target_link_libraries(icp_comparator_node ${PCL_LIBRARIES})

# --- 설치 ---
install(TARGETS
  scan_processor_node
  obstacle_detector_node
  visualization_node
  delay_monitor_node
  icp_comparator_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# --- export: 런타임에 필요한 훅/경로 노출 ---
ament_export_include_directories(include ${PCL_INCLUDE_DIRS})
ament_export_dependencies(
  rosidl_default_runtime   # ★ 메시지 런타임 의존성 내보내기
  rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs
  visualization_msgs message_filters
  tf2 tf2_ros tf2_geometry_msgs
  PCL pcl_conversions Eigen3
)

ament_package()
