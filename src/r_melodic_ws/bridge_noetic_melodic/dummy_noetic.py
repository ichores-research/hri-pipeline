#!/usr/bin/env python
import rospy
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

        # Default pose values
        self.default_pose = [0.1578, 0.2267, 0.4961, 0.72, 0.69, -0.031, -0.0638]  # x, y, z, roll, pitch, yaw

    def send_pose(self, arm, pose_values):
        """
        Sends a Pose message to the /target_pose topic.
        """
        pose = Pose()

        # Set position (x, y, z)
        pose.position.x = pose_values[0]
        pose.position.y = pose_values[1]
        pose.position.z = pose_values[2]

        # Set orientation (roll, pitch, yaw) converted to quaternion
        # We use tf.transformations.quaternion_from_euler to convert the roll, pitch, yaw to a quaternion
        quat = quaternion_from_euler(pose_values[3], pose_values[4], pose_values[5])
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        # Log and publish the pose message
        rospy.loginfo(f"Sending {arm} arm to pose: {pose}")
        self.pose_pub.publish(pose)

    def send_gripper_command(self, command):
        """
        Sends a String message to the /target_gripper topic.
        The command must be one of "00", "01", "10", "11".
        """
        if command in ["00", "01", "10", "11"]:
            rospy.loginfo(f"Sending gripper command: {command}")
            self.gripper_pub.publish(command)
        else:
            rospy.logwarn("Invalid gripper command. Please use one of '00', '01', '10', '11'.")

    def process_input(self):
        """
        Continuously reads input from the user and sends appropriate commands.
        """
        rospy.loginfo("Ready to accept commands. Type 'left <x> <y> <z> <roll> <pitch> <yaw>' or 'right <x> <y> <z> <roll> <pitch> <yaw>'.")
        rospy.loginfo("For gripper control, type 'gripper 00', 'gripper 01', 'gripper 10', or 'gripper 11'.")

        while not rospy.is_shutdown():
            try:
                user_input = input("Enter command: ").strip().split()

                if len(user_input) == 0:
                    continue  # Ignore empty inputs

                # Handling arm pose commands
                if user_input[0] in ["left", "right"]:
                    arm = user_input[0]
                    if len(user_input) == 7:  # 6D vector provided
                        pose_values = list(map(float, user_input[1:]))
                    else:
                        # Use default pose if no values are provided
                        rospy.loginfo(f"No pose values provided for {arm} arm. Using default pose.")
                        
                        pose_values = self.default_pose

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
