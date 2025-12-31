#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from your_package.msg import DeicticSolution 
from input_handling.msg import MicAnalyzerOut
from collections import deque

class DeicticBufferNode:
  def __init__(self):
    # Load parameters
    self.ros_params = rospy.get_param("deictic_buffer")
    self.input_topic = self.ros_params.get("input_topic", "analyzer_out")
    self.deictic_topic = self.ros_params.get("deictic_topic", "deictic_solutions")
    self.buffer_seconds = self.ros_params.get("buffer_seconds", 3.0)
    self.output_topic = rospy.get_param('~output_topic')

    # Initialize buffer
    self.buffer = deque()
    self.buffer_time_limit = rospy.Duration(self.buffer_seconds)

    # Subscribers
    rospy.Subscriber(self.input_topic, MicAnalyzerOut, self.input_callback)
    rospy.Subscriber(self.deictic_topic, DeicticSolution, self.deictic_callback)

    # Publisher
    self.output_pub = rospy.Publisher(self.output_topic, DeicticSolution, queue_size=10)

  def input_callback(self, msg):
    start_time = rospy.Time.from_sec(msg.start_time)  # Assuming start_time is a float
    end_time = rospy.Time.from_sec(msg.end_time)    # Assuming end_time is a float
    text = msg.text

    # Filter buffer and publish results
    self.publish_filtered_data(start_time, end_time, text)

  def deictic_callback(self, msg):
    # Add new DeicticSolution message to the buffer
    self.buffer.append(msg)
    self.clean_buffer()

  def clean_buffer(self):
    # Remove oldest messages if they exceed the buffer duration
    while self.buffer and (rospy.Time.now() - self.buffer[0].header.stamp) > self.buffer_time_limit:
      self.buffer.popleft()

  def publish_filtered_data(self, start_time, end_time, text):
    filtered_data = []
    for msg in self.buffer:
      if start_time <= msg.header.stamp <= end_time:
        filtered_data.append(msg)

    # Publish filtered data
    for data in filtered_data:
      self.output_pub.publish(data)

if __name__ == '__main__':
  rospy.init_node('deictic_buffer_node')
  node = DeicticBufferNode()
  rospy.spin()
