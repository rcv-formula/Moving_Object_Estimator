cmake_minimum_required(VERSION 3.8)
project(wall_map)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(include)

add_library(wall_map_lib
  src/wall.cpp
)

ament_target_dependencies(wall_map_lib
  ament_index_cpp
  nav_msgs
)

target_link_libraries(wall_map_lib
  yaml-cpp
)

install(TARGETS
  wall_map_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(wall_map_lib)
ament_export_dependencies(ament_index_cpp nav_msgs yaml-cpp)

ament_package()
