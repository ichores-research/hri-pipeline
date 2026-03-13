#!/usr/bin/env python

import sys
import rospy
from geometry_msgs.msg import Pose, PoseStamped
import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from moveit_commander.exception import MoveItCommanderException
from std_msgs.msg import String
import moveit_msgs.msg
import geometry_msgs.msg

def main():
    # Initialize the moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    # Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
    robot = RobotCommander()

    # Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
    scene = PlanningSceneInterface()

    # Instantiate a MoveGroupCommander object. This object is an interface to one group of joints.
    group_name = "arm"  # Change this to your group name
    move_group = MoveGroupCommander(group_name, wait_for_servers=15)

    # We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Available Planning Groups:", robot.get_group_names())

    # Misc variables
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    # Planning to a joint-space goal
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = 1.0
    joint_goal[1] = 1.0
    joint_goal[2] = 1.0
    joint_goal[3] = 1.0
    joint_goal[4] = 1.0
    joint_goal[5] = 1.0

    # Plan the trajectory
    move_group.set_joint_value_target(joint_goal)

    success = False
    # success, plan = move_group.plan()
    plan = move_group.plan()
    rospy.logwarn(str(plan))

    if success:
        print("============ Visualizing plan 1")
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        display_trajectory_publisher.publish(display_trajectory)

        # Wait for the user to confirm the trajectory
        print("============ Press 'Enter' to execute the trajectory...")
        input()

        # Execute the trajectory
        move_group.go(wait=True)

        # Clear the trajectory
        move_group.clear_pose_targets()

    else:
        print("============ Failed to plan the trajectory")

    # Wait for the user to confirm the trajectory
    print("============ Press 'Enter' to execute the trajectory...")
    input()

    # Execute the trajectory
    # move_group.go(wait=True)

    # Clear the trajectory
    move_group.clear_pose_targets()

    # Shutdown the moveit_commander
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
