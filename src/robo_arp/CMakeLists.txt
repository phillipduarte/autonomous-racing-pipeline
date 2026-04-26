cmake_minimum_required(VERSION 3.8)
project(robo_arp)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the Python package (robo_arp/ module)
ament_python_install_package(${PROJECT_NAME})

# Install node executables into lib/<pkg>/ so `ros2 run` can find them
install(PROGRAMS
  scripts/coordinator_node
  scripts/slam_monitor_node
  scripts/safety_monitor_node
  scripts/wall_follower_wrapper_node
  scripts/map_processor_node
  scripts/pure_pursuit_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
