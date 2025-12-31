#!/usr/bin/python

import rospy
import zmq
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Set up ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

# ROS to OpenCV image converter
bridge = CvBridge()

def callback(color_msg, depth_msg):
    # Convert ROS Image message to OpenCV format for color
    color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    # Convert ROS Image message to OpenCV format for depth
    depth_image = bridge.imgmsg_to_cv2(depth_msg, "16UC1")  # Depth is usually 16-bit

    # Encode color image as JPEG
    _, color_buffer = cv2.imencode('.jpg', color_image)
    # Encode depth image as PNG to preserve depth information
    _, depth_buffer = cv2.imencode('.png', depth_image)

    # Combine both images into a single message
    # Prefix each image with its length for parsing
    message = (
        np.int32(len(color_buffer)).tobytes() + color_buffer.tobytes() +
        np.int32(len(depth_buffer)).tobytes() + depth_buffer.tobytes()
    )

    # Send the combined message over ZeroMQ
    socket.send(message)

# Initialize ROS node
rospy.init_node('rgbd_publisher')

# Subscribers for color and depth images
color_sub = Subscriber('/xtion/rgb/image_raw', Image)
depth_sub = Subscriber('/xtion/depth/image_raw', Image)

# Synchronize the two topics
ats = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
ats.registerCallback(callback)

rospy.spin()