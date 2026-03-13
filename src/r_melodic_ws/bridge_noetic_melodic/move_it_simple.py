#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose, PoseStamped
import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from moveit_commander.exception import MoveItCommanderException
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from moveit_msgs.msg import CollisionObject,
from shape_msgs.msg import Plane
import moveit_msgs.msg

class Mover():
    def __init__(self):
        # moveit_commander.roscpp_initialize([])
        rospy.init_node('pose_subscriber_node', anonymous=True)

        # Initialize RobotCommander to get the robot's arm groups
        self.robot = RobotCommander()

        self.scene = PlanningSceneInterface()


        # Get the group names of the robot
        self.groups = self.robot.get_group_names()
        # loginfo("Available Move Groups: %s" % self.groups)
        rospy.loginfo("Available Move Groups: %s", self.groups)


        box_name = "my_awesome_box"
        # box_pose = PoseStamped()
        # box_pose.header.frame_id = "base_link"
        # box_pose.pose.orientation.w = 1.0
        # box_pose.pose.position.z = 0.11  # above the panda_hand frame
        # self.scene.add_box(box_name, box_pose, size=(0.075, 0.075, 0.075))
        # rospy.spin()
        # self.scene.remove_world_object("my_awesome_box")
        # rospy.logwarn(self.scene.apply_planning_scene.__doc__)
        found_objects = self.scene.get_objects(['my_awesome_box'])
        rospy.loginfo(str(found_objects))

        # cube = found_objects['my_awesome_box']
        # cube.pose.position.x = 0.5
        # self.scene.add_box(box_name, cube, size=(0.075, 0.075, 0.075))
        # rospy.loginfo(str(self.scene.get_objects()))
        self.scene.remove_world_object("my_awesome_box")



        # Create a plane collision object
        plane_pose = Pose()
        plane_pose.orientation.w = 1.0
        plane_pose.position.x = 0.0
        plane_pose.position.y = 0.0
        plane_pose.position.z = 0.0

        plane = CollisionObject()
        plane.id = "plane"
        plane.header.frame_id = "world"
        plane.primitives = [moveit_msgs.msg.Shape()]
        plane.primitives[0].type = moveit_msgs.msg.Shape.PLANE
        plane.primitives[0].plane.coef.x = 0.0
        plane.primitives[0].plane.coef.y = 0.0
        plane.primitives[0].plane.coef.z = 1.0
        plane.primitives[0].plane.coef.w = 0.0

        plane.primitive_poses = [plane_pose]
        plane.operation = moveit_msgs.msg.CollisionObject.ADD

        # Add the plane to the planning scene
        scene.add_object(plane)

        co = CollisionObject()

        co.id = "floor2"
        co.header.frame_id = "base_link"

        plane = Plane()
        plane.coef = [0.0, 0.0, 1.0, 0.5]  # z=0 plane equation

        co.planes = [plane]
        co.operation = CollisionObject.ADD

        planning_scene_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)
        planning_scene_pub.publish(co)
        rospy.spin()

        # moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    mover = Mover()
