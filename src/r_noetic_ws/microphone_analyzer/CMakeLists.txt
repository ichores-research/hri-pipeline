cmake_minimum_required(VERSION 3.0.2)
project(microphone_analyzer)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  audio_common_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/vad_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
