cmake_minimum_required(VERSION 3.8)
project(robo_arp_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SaveMap.srv"
  "srv/SetActive.srv"
  "srv/ProcessMap.srv"
  "action/RunLap.action"
)

ament_package()
