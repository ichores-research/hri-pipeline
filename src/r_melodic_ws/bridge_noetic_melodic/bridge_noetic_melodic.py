#!/usr/bin/env python

# Script for bridging from ROS 1 noetic to ROS 1 melodic action messages.
# It creates node for each functionality:
#  - move arm to position (Pose)
#  - others
# author: Simon Dratva <dratvsim@fel.cvut.cz>
# part of ichores project at CIIRC

import rospy
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, RobotCommander
from moveit_commander.exception import MoveItCommanderException
from std_msgs.msg import String

bridge_logs_publisher = rospy.Publisher('/bridge_logs', String, queue_size=10)

def loginfo(msg):
    """
    Custom function for publishing the loginfo message
    to /bridge_logs topic
    """
    rospy.loginfo(msg)
    bridge_logs_publisher.publish(msg)

def logwarn(msg):
    """
    Custom function for publishing the logwarn message
    to /bridge_logs topic
    """
    rospy.logwarn(msg)
    bridge_logs_publisher.publish(msg)

def logerr(msg):
    """
    Custom function for publishing the logerr message
    to /bridge_logs topic
    """
    rospy.logerr(msg)
    bridge_logs_publisher.publish(msg)

class PoseSubscriber:
    def __init__(self):
        # rospy.init_node('pose_subscriber_node', anonymous=True)

        # Initialize RobotCommander to get the robot's arm groups
        self.robot = RobotCommander()

        # Get the group names of the robot
        self.groups = self.robot.get_group_names()
        # loginfo("Available Move Groups: %s" % self.groups)
        rospy.loginfo("Available Move Groups: %s", self.groups)

        if "arm_torso" in self.groups:
            # Create a mapping for the arms based on the group names
            self.arm_groups = {
                'arm': self.find_arm_group('arm_torso')
            }
        else:
            # Create a mapping for the arms based on the group names
            self.arm_groups = {
                'left': self.find_arm_group('left_torso'),
                'right': self.find_arm_group('right_torso')
            }

        if not all(self.arm_groups.values()):
            rospy.logerr("Arm groups not found. Check your robot model and group names.")
            exit(1)

        # Set tolerances for each group
        for group_name in self.arm_groups.values():
            group = MoveGroupCommander(group_name)
            group.set_pose_reference_frame("base_link")
            group.set_goal_position_tolerance(0.001)
            group.set_goal_orientation_tolerance(0.001)

        rospy.Subscriber("/target_pose", Pose, self.pose_callback)
        self.moving = False

        rospy.loginfo("Pose Subscriber initialized. Waiting for target poses...")

    def find_arm_group(self, arm_side):
        """
        Find the correct arm group based on the side ('left' or 'right').
        This assumes that group names contain the side information, such as 'arm_left' and 'arm_right'.
        """
        for group in self.groups:
            if arm_side in group:
                return group
        return None

    def pose_callback(self, pose):
        """
        Callback function for Pose Subscriber
        """
        arm = 'right'
        if pose.position.y >= 0.0:
            # this logic for determining which hand should be moved could(should) be enhanced
            arm = 'left'

        if not self.moving:
            rospy.loginfo("Received new target pose for %s arm." % arm)
            self.move_to_pose(arm, pose)
        else:
            loginfo("FAILED: Robot is already moving. Ignoring new pose.")

    def move_to_pose(self, arm, target_pose, end=False):
        """
        Function that sends given position to MoveIt commander
        """
        try:
            self.moving = True
            group_name = self.arm_groups.get(arm)
            if not group_name:
                logwarn("FAILED: No move group found for %s arm." % arm)
                return
            # if target_pose == [1 2 3 4 5 6 7]:
            #     target_pose = TODO add joint move with home position

            group = MoveGroupCommander(group_name)
            group.set_pose_target(target_pose)

            loginfo("MOVING: Moving %s arm to target pose" % arm)
            plan = group.go(wait=True)

            if plan:
                loginfo("SUCCESS: Successfully moved %s arm to the target pose." % arm)
            else:
                logwarn("FAILED: Failed to move %s arm to the target pose." % arm)
                # if end:
                #     rospy.logwarn("Failed to move %s arm to the target pose." % arm)
                # else:
                #     rospy.logwarn("Failed to move %s arm to the target pose.\nTrying the other arm" % arm)
                #     self.move_to_pose('left' if arm=='right' else 'right', target_pose, end=True)

            group.stop()
            group.clear_pose_targets()

        except MoveItCommanderException as e:
            logerr("FAILED: MoveIt Commander Exception: {}".format(e))

        finally:
            self.moving = False

class GripperSubscriber:
    def __init__(self):
        # rospy.init_node('gripper_subscriber_node', anonymous=True)

        # Initialize RobotCommander to get robot gripper groups
        self.robot = RobotCommander()

        # Get group names of the robot
        self.groups = self.robot.get_group_names()

        # Mapping for the grippers on the group names
        self.gripper_groups = {
            'left': self.find_group('gripper_left'),
            'right': self.find_group('gripper_right')
        }

        if not all(self.gripper_groups.values()):
            rospy.logerr("Gripper groups not found. Check your robot model and group names.")
            exit(1)

        rospy.Subscriber("/target_gripper", String, self.grip_callback)
        self.moving = False

        rospy.loginfo("Gripper Subscriber initialized. Waiting for target poses...")

    def find_group(self, name):
        """
        Find the correct arm group based on the side ('left' or 'right').
        This assumes that group names contain the side information, such as 'arm_left' and 'arm_right'.
        """
        for group in self.groups:
            if name in group:
                return group
        return None

    def grip_callback(self, data):
        """
        Callback function for Gripper Subscriber,
        input: '00', '01', '10' or '11', where 0 is open and 1 close,
        left digit for left gripper and respectively the right digit
        """
        message = data.data.strip()

        if message not in ['00', '01', '10', '11']:
            logerr("Wrong input to gripper controller ['00', '01', '10' or '11']")
            return

        rospy.loginfo("Received gripper command: %s" % message)

        if not self.moving:
            self.execute_gripper_command(message)
        else:
            rospy.loginfo("Gripper is already moving. Ignoring new command.")

    def execute_gripper_command(self, command):
        try:
            self.moving = True

            # Mapping the command to joint values
            joint_open = 0.043
            joint_closed = 0.001

            left_joint_val = joint_closed if command[0] == '1' else joint_open
            right_joint_val = joint_closed if command[1] == '1' else joint_open

            loginfo("MOVING: Setting left gripper to value: %f, right gripper to value: %f" % (left_joint_val, right_joint_val))

            # Moving grippers to values from command
            self.move_gripper('left', left_joint_val)
            self.move_gripper('right', right_joint_val)

            loginfo("SUCCESS: Both grippers moved to desired positions")
        except MoveItCommanderException as e:
            logerr("FAILED: MoveIt Commander Exception: {}".format(e))

        finally:
            self.moving = False

    def move_gripper(self, arm, joint_value):
        """
        Moves both joints of the specified gripper to given joint value
        """
        group_name = self.gripper_groups.get(arm)
        if not group_name:
            logwarn("No gripper group found for %s arm" % arm)
            return

        group = MoveGroupCommander(group_name)

        # Set the target joint values for both joints
        joint_goal = group.get_current_joint_values()
        rospy.loginfo("joint_goal: {}".format(joint_goal))

        if len(joint_goal) >= 2:
            joint_goal[0] = joint_value
            joint_goal[1] = joint_value
        else:
            rospy.logwarn("Expected 2 joints for %s gripper, but got %d joints" % (arm, len(joint_goal)))

        group.go(joint_goal, wait=True)

        rospy.loginfo("Succesfully moved %s gripper to joint value %f" % (arm, joint_value))

        group.stop()


if __name__ == '__main__':
    try:
        rospy.init_node('pose_gripper_bridge', anonymous=True)
        PoseSubscriber()
        GripperSubscriber()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
