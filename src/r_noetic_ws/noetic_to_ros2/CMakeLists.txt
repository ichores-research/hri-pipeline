cmake_minimum_required(VERSION 3.0.2)
project(noetic_to_ros2)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)
catkin_python_setup()

add_message_files(
  FILES
  HRICommand.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

install(PROGRAMS
  src/noetic_to_ros2/zmq_rs_publisher.py
  src/noetic_to_ros2/zmq_tf_publisher.py
  src/noetic_to_ros2/zmq_gdrnet_publisher.py
  src/noetic_to_ros2/zmq_hri_subscriber.py
  
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

