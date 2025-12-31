#!/usr/bin/python

import rospy
import time
import zmq
import cv2
import struct
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf.listener import TransformListener
from tf import LookupException, ConnectivityException, ExtrapolationException

# Set up ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

# Initialize ROS node
rospy.init_node('tf_publisher')

# Set up the TransformListener
tf_listener = TransformListener()

def publish_transform():
    try:
        translation, rotation = tf_listener.lookupTransform("/base_footprint", "/xtion_rgb_optical_frame", rospy.Time(0))
        # Convert translation and rotation into a numpy array
        tf_data = np.array([*translation, *rotation])
        
        # Print the transformation for debugging purposes
        rospy.loginfo(f"Transform: {tf_data}")
        
        # Pack the data and send it over ZeroMQ
        packed_data = struct.pack(f"{len(tf_data)}f", *tf_data)
        socket.send(packed_data)
        
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        rospy.logerr(f"Error looking up transform: {e}")

# Loop with rospy Timer for better control in a ROS environment
def timer_callback(event):
    publish_transform()

# Set up a ROS Timer to publish at a specified rate
publish_rate = 1.0  # Hz
rospy.Timer(rospy.Duration(1.0 / publish_rate), timer_callback)

# Keep the node running
rospy.spin()

