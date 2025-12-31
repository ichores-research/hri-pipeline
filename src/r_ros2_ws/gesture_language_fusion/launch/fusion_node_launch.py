#!/usr/bin/env python
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gesture_language_fusion',
            executable='fusion_node',
            name='gesture_language_fusion_node',
            output='screen',
        ),
    ])


