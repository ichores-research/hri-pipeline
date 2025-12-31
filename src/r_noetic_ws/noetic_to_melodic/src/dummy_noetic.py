#!/usr/bin/env python
import rospy
import numpy as np
import json

from geometry_msgs.msg import Pose
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler

class CommandSender:
    def __init__(self):
        # Initialize the node
        rospy.init_node('command_sender_node', anonymous=True)
        
        # Publishers for target_pose and target_gripper topics
        self.pose_pub = rospy.Publisher('/target_pose', Pose, queue_size=10)
        self.gripper_pub = rospy.Publisher('/target_gripper', String, queue_size=10)

        self.grasps_sub = rospy.Subscriber('/gdrnet_grasps', String, self.grasps_callback)
        self.grasps = None

        # Default pose values
        self.default_pose = [0.5, -0.5, 1.0, 0.0, -1.57, 1.57]  # x, y, z, roll, pitch, yaw
        self.arm_home_joint_values = [-1.10, 1.47, 2.71, 1.71, -1.57, 1.39, 0.00] # same for both arms

        rospy.sleep(1) # DO NOT REMOVE OR DISASTER HAPPENS!!!!!!

    def grasps_callback(self, msg):
        self.grasps = json.loads(str(msg.data))
        return 

    def offset_pose(self, pose, offset):
        """
        Calculates new pose with same orientation but moved forward by offset
        Parameters:
            pose (array-like): [x, y, z, roll, pitch, yaw]
                            x, y, z in meters
                            roll, pitch, yaw in radians
        Returns:
            np.ndarray: [x_new, y_new, z_new, roll, pitch, yaw]
            or None if bad input
        """
        def calculate_R_from_euler(roll, pitch, yaw):
            # Returns Rx, Ry, Rz if pose with euler orientation
            Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
                ])

            # Rotation about Y-axis (pitch)
            Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
                    ])

            # Rotation about Z-axis (yaw)
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
                    ])

            # Combined rotation matrix R = Rz * Ry * Rx
            R = Rz @ Ry @ Rx
            return R

        def calculate_R_from_quat(qx, qy, qz, qw):
            # Returns Rx, Ry, Rz if pose with quaternion orientation
            # Precompute squares
            qx2 = qx * qx
            qy2 = qy * qy
            qz2 = qz * qz
            
            # Compute rotation matrix elements
            R = np.array([
                [1 - 2*(qy2 + qz2),    2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),    1 - 2*(qx2 + qz2),   2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),    2*(qy*qz + qx*qw),   1 - 2*(qx2 + qy2)]
            ], dtype=float)
            
            return R
        # Create Pose()
        ret_pose = Pose()

        if len(pose) == 6:
            # euler
            roll, pitch, yaw = pose[3:]
            R = calculate_R_from_euler(roll, pitch, yaw)
            quat = quaternion_from_euler(roll, pitch, yaw)
            ret_pose.orientation.x = quat[0]
            ret_pose.orientation.y = quat[1]
            ret_pose.orientation.z = quat[2]
            ret_pose.orientation.w = quat[3]
        elif len(pose) == 7:
            qx, qy, qz, qw = pose[3:]
            R = calculate_R_from_quat(qx, qy, qz, qw)
            ret_pose.orientation.x = qx
            ret_pose.orientation.y = qy
            ret_pose.orientation.z = qz
            ret_pose.orientation.w = qw
        else:
            R = None
            rospy.logwarn("Bad Pose given")
            return R
        
        # Unpack the input pose
        x, y, z = pose[:3]
        dir = np.array([1, 0, 0])

        # Transform the forward vector into world coordinates
        dir_world = R @ dir

        # Move forward by 0.25 meters
        displacement = offset * dir_world
        ret_pose.position.x = x + displacement[0]
        ret_pose.position.y = y + displacement[1]
        ret_pose.position.z = z + displacement[2]

        # Return the new pose with updated position
        return ret_pose

    def send_pose(self, arm, pose_values, offset=-0.25):
        """
        Sends a Pose message to the /target_pose topic.
        """
        # default offest -0.25 PAL gripper constant
        pose = self.offset_pose(pose_values, offset)

        if pose != None:
            # Log and publish the pose message
            rospy.loginfo(f"Sending {arm} arm to pose: {pose}")
            self.pose_pub.publish(pose)

    def send_gripper_command(self, command):
        """
        Sends a String message to the /target_gripper topic.
        The command must be one of "00", "01", "10", "11".
        """
        # rospy.sleep(1)
        if command in ["00", "01", "10", "11"]:
            rospy.loginfo(f"Sending gripper command: {command}")
            self.gripper_pub.publish(command)
        else:
            rospy.logwarn("Invalid gripper command. Please use one of '00', '01', '10', '11'.")

    def process_input(self):
        """
        Continuously reads input from the user and sends appropriate commands.
        """
        rospy.loginfo("""Ready to accept commands.\n For controlling arms type: \n\t'left/right <x> <y> <z> <roll> <pitch> <yaw>' or \n
                      'left/right <x> <y> <z> <qx> <qy> <qz> <qw>'""")
        rospy.loginfo("""For gripper control type: (1 ~ close, 0 ~ open)\n\t
                      'gripper 00', 'gripper 01', 'gripper 10', or 'gripper 11'""")

        while not rospy.is_shutdown():
            try:
                if self.grasps:
                    print("Grasps loaded from GDRNet, to see first one, press Enter")
                user_input = input("Enter command: ").strip().split()

                if len(user_input) == 0:
                    idx = input(f"Enter idx of grasp you want to see in range(0,{len(self.grasps)-1}): ")
                    try:
                        idx = int(idx)
                    except:
                        idx = 0
                        print("Inputed not a number, so first is shown")
                    print(f"Grasp id:{self.grasps[idx]['id']}")
                    # print(f"Position:\n{self.grasps[0]['position']}")
                    # print(f"Orientation:\n{self.grasps[0]['orientation']}")
                    cmd_x = self.grasps[idx]['position'][0]
                    cmd_y = self.grasps[idx]['position'][1]
                    cmd_z = self.grasps[idx]['position'][2]

                    cmd_qx = self.grasps[idx]['orientation'][0]
                    cmd_qy = self.grasps[idx]['orientation'][1]
                    cmd_qz = self.grasps[idx]['orientation'][2]
                    cmd_qw = self.grasps[idx]['orientation'][3]
                    print(f"That can be tested by entering command:\nright {cmd_x} {cmd_y} {cmd_z} {cmd_qx} {cmd_qy} {cmd_qz} {cmd_qw}")
                    continue  # Ignore empty inputs
                
                # Handling arm pose commands
                if user_input[0] == "home":
                    # send magic Pose that bridge node recognises as command to home arms
                    rospy.logwarn(f"Homing both arms")
                    self.send_pose(arm, [1,2,3,4,5,6,7])
                if user_input[0] in ["left", "right"]:
                    arm = user_input[0]
                    if len(user_input) == 7 or len(user_input) == 8:  # 6D vector provided
                        pose_values = list(map(float, user_input[1:]))
                    else:
                        # Use default pose if no values are provided
                        rospy.loginfo(f"No pose values provided for {arm} arm. Using default pose.")
                        if user_input[0] == "left":
                            pose_values = self.default_pose
                        elif user_input[0] == "right":
                            pose_values = self.default_pose
                            pose_values[1] *= -1

                    # Send the pose command to the arm
                    self.send_pose(arm, pose_values)

                # Handling gripper commands
                elif user_input[0] == "gripper" and len(user_input) == 2:
                    gripper_command = user_input[1]
                    self.send_gripper_command(gripper_command)
                # Stopping the input stream
                elif user_input[0] in ["end", "exit", "stop", "quit", "q"]:
                    rospy.signal_shutdown("end")
                    break

                else:
                    rospy.logwarn("Invalid command format. Please try again.")

            except Exception as e:
                rospy.logerr(f"Error processing input: {e}")


if __name__ == "__main__":
    try:
        # Initialize and run the command sender
        sender = CommandSender()
        sender.process_input()
    except rospy.ROSInterruptException:
        pass
