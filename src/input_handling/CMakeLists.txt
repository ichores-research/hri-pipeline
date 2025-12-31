cmake_minimum_required(VERSION 3.0.2)
project(input_handling)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  audio_common_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  MicAnalyzerOut.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs audio_common_msgs geometry_msgs
)

catkin_install_python(PROGRAMS
  src/deictic_buffer_node.py
  src/fusion_node.py
  src/mic_analyzer_node.py
  src/mic_publisher_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)