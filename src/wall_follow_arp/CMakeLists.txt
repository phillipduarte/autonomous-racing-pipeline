cmake_minimum_required(VERSION 3.8)
project(wall_follow_arp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)

add_executable(wall_follow_node wall_follow.cpp)
ament_target_dependencies(wall_follow_node rclcpp sensor_msgs ackermann_msgs)

install(TARGETS wall_follow_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
