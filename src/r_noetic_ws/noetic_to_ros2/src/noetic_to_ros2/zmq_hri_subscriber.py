#!/usr/bin/env python3

import rospy, zmq, json
from std_msgs.msg import Header
from noetic_to_ros2.msg import HRICommand

TIMEOUT_MS = 1000

class ZMQSubscriber:
  def __init__(self):
    rospy.init_node('zmq_hri_subscriber_node')
    
    # ZeroMQ context and socket setup
    context = zmq.Context()
    self.socket = context.socket(zmq.SUB)
    self.socket.connect("tcp://localhost:5558")
    self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    self.poller = zmq.Poller()
    self.poller.register(self.socket, zmq.POLLIN)
    
    self.rate = rospy.Rate(1) # 1 Hz
    
    # ROS publisher to output topic
    self.pub = rospy.Publisher('/reasoner/hricommand', HRICommand, queue_size=10)
  
  def run(self):
    rospy.loginfo("HRI Subscriber node started")
    
    while not rospy.is_shutdown():
      try:
        # Receive and parse message
        socks = dict(self.poller.poll(TIMEOUT_MS))

        if self.socket in socks:
          msg_str = self.socket.recv_string()
        
          # Reconstruct HRICommand from received message
          msg = HRICommand()
          msg.header = Header()
          msg.header.stamp = rospy.Time.now()
          msg.header.frame_id = "reasoner"
          msg.data = [msg_str]

          # Publish to ROS topic
          self.pub.publish(msg)
          rospy.loginfo("Message published to /reasoner/hricommand")

        else:
          rospy.loginfo("Waiting for message ...")

        self.rate.sleep()

      except Exception as e:
        rospy.logerr(f"Error receiving message: {e}")
        

if __name__ == '__main__':
  subscriber = ZMQSubscriber()
  try:
    subscriber.run()
  except rospy.ROSInterruptException:
    rospy.loginfo("Node terminated")