#!/usr/bin/env python
import numpy as np
import threading, json, os, yaml, zmq
import tf.transformations as tf_trans

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from noetic_to_ros2.msg import HRICommand

MOVEMENT_STATUS_SUCCESS_KEYWORD = "SUCCESS"
MOVEMENT_STATUS_FAILED_KEYWORD = "FAILED"
MOVEMENT_STATUS_MOVING_KEYWORD = "MOVING"

MOVEMENT_STATUS_FAILED = "failed"
MOVEMENT_STATUS_SUCCESS = "success"
MOVEMENT_STATUS_MOVING = "moving"
MOVEMENT_STATUS_WAITING = "waiting"
MOVEMENT_STATUS_CANCELED = "canceled"

GRIPPER_BOTH_OPEN = "00"
GRIPPER_BOTH_CLOSED = "11"
GRIPPER_OPEN_LEFT = "01"
GRIPPER_OPEN_RIGHT = "10"

DEFAULT_TABLE_HEIGHT = 0.50


class InputManager():
    def __init__(self):
        self.grasp_annotations = self.load_grasp_annotations('src/noetic_to_melodic/src/annotations_tiago', 'src/noetic_to_melodic/src/ycb_ichores.yaml')
        self.data_loaded = False

        self.hri_sub = rospy.Subscriber('/reasoner/hricommand', HRICommand, self.hri_callback)
        rospy.loginfo("InputManager inicialized")
        pass
    
    def hri_callback(self, msg):
        self.hricommand = json.loads(msg.data[0])
        rospy.loginfo(f"Received HRICommand for action: {self.hricommand['action']}")
        if self.load_gdrnet():
            self.data_loaded = True
        return

    def wait_for_data(self):
        while not rospy.is_shutdown() and not self.data_loaded:
            rospy.logwarn("Waiting for data from reasoner... ")
            rospy.sleep(0.2) # 1

        return self.data_loaded

    def load_gdrnet(self):
        try:
            msg = rospy.wait_for_message('/gdrnet_object_poses', String, timeout=5)
        except rospy.ROSException as e:
            rospy.logerr(f"[load_gdrnet] Error in wait_for_message: {e}")
            return False
        
        self.gdrn_objects = json.loads(msg.data)
        return True

    def load_grasp_annotations(self, folder_path, yaml_file_path):
        with open(yaml_file_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        names = yaml_data.get('names', {})
        grasp_annotations = {}

        for obj_id, obj_name in names.items():
            filename = f"obj_{int(obj_id):06d}.npy"
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                grasps = np.load(file_path)
                grasp_annotations[obj_name] = {'grasps': grasps}
        
        return grasp_annotations
    
    def load_graps(self, obj_name):
        grasps = self.grasp_annotations.get(obj_name[:-2], None)['grasps']
        return grasps
    
    def transform_grasps(self, grasps_obj_frame, obj):
        def transform_grasp_obj2world(grasps, pose):
            transformed_grasps = []

            # Convert object quaternion to a 4x4 transformation matrix, then extract the 3x3 rotation matrix
            obj_quat = pose[3:]
            obj_transform = tf_trans.quaternion_matrix(obj_quat)  # This gives a 4x4 matrix
            obj_R = obj_transform[:3, :3]  # Extract the 3x3 rotation part
            obj_t = np.array(pose[:3])  # Translation vector

            align_x_to_z = tf_trans.quaternion_from_euler(0, np.pi / 2, 0)
            for grasp in grasps:
                # Convert the grasp to a 4x4 matrix
                grasp_matrix = np.array(grasp).reshape(4, 4)

                # Apply rotation and translation to transform the grasp to world coordinates
                transformed_grasp_matrix = np.eye(4)
                transformed_grasp_matrix[:3, :3] = np.dot(obj_R, grasp_matrix[:3, :3])  # Rotate
                transformed_grasp_matrix[:3, 3] = np.dot(obj_R, grasp_matrix[:3, 3]) + obj_t  # Rotate and translate

                position = transformed_grasp_matrix[:3, 3]
                orientation_quat = tf_trans.quaternion_from_matrix(transformed_grasp_matrix)
                adjusted_orientation = tf_trans.quaternion_multiply(orientation_quat, align_x_to_z)
                
                transformed_grasps.append([*position, *adjusted_orientation])

            return transformed_grasps

        grasps = []
        obj_pose = [obj['position'][0], obj['position'][1], obj['position'][2],
                    obj['orientation'][0], obj['orientation'][1], obj['orientation'][2], obj['orientation'][3] ]
        
        grasps = transform_grasp_obj2world(grasps_obj_frame, obj_pose)
        # print(grasps)

        return grasps


class CommandSender:
    def __init__(self):
        # Publishers for target_pose and target_gripper topics
        self.pose_pub = rospy.Publisher('/target_pose', Pose, queue_size=10)
        self.gripper_pub = rospy.Publisher('/target_gripper', String, queue_size=10)

        self.grasps_sub = rospy.Subscriber('/gdrnet_grasps', String, self.grasps_callback)
        self.grasps = None

        # Default pose values
        self.default_pose = [0.5, -0.5, 1.0, 0.0, -1.57, 1.57]  # x, y, z, roll, pitch, yaw

        rospy.sleep(1) # DO NOT REMOVE OR DISASTER HAPPENS!!!!!!
        rospy.loginfo("CommandSender inicialized")

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
            quat = tf_trans.quaternion_from_euler(roll, pitch, yaw)
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
            rospy.loginfo("Sending %s arm to pose: [%.3f %.3f %.3f %.3f %.3f %.3f %.3f]" % 
                          (arm,pose.position.x, pose.position.y,pose.position.z, 
                           pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
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


class ActionTemplate():
    """
    Class representing the action to be executed by the robot
    """
    def __init__(
        self,
        action_type,
        command_sender,
        confirm_poses=True
    ):
        self.action_type = action_type
        self.command_sender = command_sender
        self.movement_status = MOVEMENT_STATUS_WAITING

        # Flag for manual confirmation of grasp and place poses finish
        self.confirm_poses = confirm_poses

        # Predefined safe poses for robot arms 
        self.arm_safe_poses = {
            'left': [0.2, 0.7, 1.0, 0, 0.707, 0.0], # TODO: not tested yet
            'right': [0.2, -0.7, 1.0, 0, 0.707, 0.0]
        }

        # create a feedback from the bridge about the motion execution results
        self.lock = threading.Lock()
        self.bridge_logs_thread = threading.Thread(target=self.bridge_logs_listener)
        self.bridge_logs_thread.start()

        # Prague tiago pose offsets
        self.x_off = 0.025
        self.y_off = 0.0

        # offsets for vertical movement 
        self.z_place_off = 0.10
        self.z_pick_off = 0.12

    def bridge_logs_listener(self):
        rospy.Subscriber("/bridge_logs", String, self.bridge_logs_callback)

    def bridge_logs_callback(self, msg):
        """
        Callback function for the logs from bridge
        It acts as a feedback from the bridge about the status of robot movement
        """
        self.lock.acquire()
        data = msg.data.strip()
        rospy.loginfo("Received: '%s'" % data)
        if MOVEMENT_STATUS_FAILED_KEYWORD in data:
            self.movement_status = MOVEMENT_STATUS_FAILED

        elif MOVEMENT_STATUS_MOVING_KEYWORD in data:
            self.movement_status = MOVEMENT_STATUS_MOVING

        elif MOVEMENT_STATUS_SUCCESS_KEYWORD in data:
            self.movement_status = MOVEMENT_STATUS_SUCCESS

        rospy.loginfo("Movement status: '%s'" % self.movement_status)
        self.lock.release()

    def get_movement_status(self):
        """
        Returns the current movemement status
        """
        self.lock.acquire()
        movement_status = self.movement_status
        self.lock.release()
        return movement_status
    
    def set_movement_status(self, status):
        """
        Sets the current movement status manually
        """
        self.lock.acquire()
        self.movement_status = status
        self.lock.release()

    def wait_for_movement_finish(self):
        """
        Simple waiting function for response from bridge
        """
        status = self.get_movement_status()
        while status == MOVEMENT_STATUS_MOVING:
            rospy.sleep(0.5)
            rospy.loginfo("Ongoing movement")
            status = self.get_movement_status()

        if self.movement_status != MOVEMENT_STATUS_SUCCESS:
            rospy.logerr("Movement failed!")
            return
        rospy.loginfo("Movement successful!")

    def parse_target_pose(self, target_pose):
        """
        Method for parsing the target pose to separate variables for clarity
        Target pose is comprised of 3 position and 4 orientation (quat) values
        """
        return (
            target_pose[0],
            target_pose[1],
            target_pose[2],
            target_pose[3],
            target_pose[4],
            target_pose[5],
            target_pose[6],
        )

    def execute_action_pick(self, target_pose):
        """
        Method containing the sequence of hand movements to execute
        pick action in given target pose
        """
        # First, we parse the target pose to separate variables
        (
            target_x,
            target_y,
            target_z,
            target_qx,
            target_qy,
            target_qz,
            target_qw,
        ) = self.parse_target_pose(target_pose)

        x_off = self.x_off
        y_off = self.y_off
        z_off = self.z_pick_off

        # Step 1: Open gripper to be ready for picking an object
        # Based on the target pose, the arm is simply chosen according to the y-axis value
        if target_pose[1] <= 0:
            gripper_command = str(GRIPPER_OPEN_RIGHT)
            arm = 'right'
        else:
            gripper_command = str(GRIPPER_OPEN_LEFT)
            arm = 'left'
        
        arm_safe_pose = self.arm_safe_poses[arm]

        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_gripper_command(gripper_command)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 1 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 2: Put the arm high to avoid collision with table
        pose = arm_safe_pose
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose("any", pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 2 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 3: Move above the object ~ pre-grasp vector
        pose = [target_x+x_off, target_y+y_off, target_z, target_qx, target_qy, target_qz, target_qw]
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose("any", pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 3 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED
        
        # If confirm_poses True, let the user confirm the safety of grabbing the objects
        if self.confirm_poses:
            confirmation = input("Is it safe to grab the object? y/n: ")
            if not (confirmation == 'y' or confirmation == 'yes'):
                return MOVEMENT_STATUS_CANCELED

        # Step 4: move to the object position (grasp pose)
        pose = [target_x+x_off, target_y+y_off, target_z-z_off, target_qx, target_qy, target_qz, target_qw]
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose("any", pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 4B finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 5: close the gripper
        gripper_command = GRIPPER_BOTH_CLOSED
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_gripper_command(gripper_command)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 5 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 6: ad Step 3
        pose = [target_x+x_off, target_y+y_off, target_z, target_qx, target_qy, target_qz, target_qw]
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose("any", pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 6 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 7: ad Step 2
        pose = arm_safe_pose
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose("any", pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 7 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        self.set_movement_status(MOVEMENT_STATUS_WAITING)

        return MOVEMENT_STATUS_SUCCESS
    
    def execute_action_place(self, target_pose):
        """
        Function handling the placement of object held to target_pose place.
        """
        # First, we parse the target pose to separate variables
        (
            target_x,
            target_y,
            target_z,
            target_qx,
            target_qy,
            target_qz,
            target_qw,
        ) = self.parse_target_pose(target_pose)

        x_off = self.x_off
        y_off = self.y_off
        z_off = self.z_place_off

        # Determine the arm to perform the movement
        if target_pose[1] <= 0:
            gripper_command = str(GRIPPER_OPEN_RIGHT)
            arm = 'right'
        else:
            gripper_command = str(GRIPPER_OPEN_LEFT)
            arm = 'left'
        
        arm_safe_pose = self.arm_safe_poses[arm]

        # Step 1: Put the arm high to avoid collision with table
        pose = arm_safe_pose
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose(arm, pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 1 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 2: Move above the target place
        pose = [target_x+x_off, target_y+y_off, target_z, target_qx, target_qy, target_qz, target_qw]
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose("any", pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 2 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED
        
        # If confirm_poses let the user confirm the safety of placing the object
        if self.confirm_poses:
            confirmation = input("Is it safe to place the object here? y/n: ")
            if not (confirmation == 'y' or confirmation == 'yes'):
                return MOVEMENT_STATUS_CANCELED
        
        # Step 3: move to the object position (grasp pose)
        pose = [target_x+x_off, target_y+y_off, target_z-z_off, target_qx, target_qy, target_qz, target_qw] 
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose(arm, pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 3B finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 4: Open the gripper
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_gripper_command(gripper_command) # from the start of the function
        self.wait_for_movement_finish()
        rospy.loginfo("Step 4 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 5: ad Step 2
        pose = [target_x+x_off, target_y+y_off, target_z, target_qx, target_qy, target_qz, target_qw]
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose(arm, pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 5 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        # Step 6: ad Step 1
        pose = arm_safe_pose
        self.set_movement_status(MOVEMENT_STATUS_MOVING)
        self.command_sender.send_pose(arm, pose)
        self.wait_for_movement_finish()
        rospy.loginfo("Step 6 finished")
        if self.get_movement_status() == MOVEMENT_STATUS_FAILED:
            return MOVEMENT_STATUS_FAILED

        self.set_movement_status(MOVEMENT_STATUS_WAITING)

        return MOVEMENT_STATUS_SUCCESS

    def execute_action_move(self, target_pose, place_pose=None):
        """
        Handles both the picking the object at target_pose 
        and placing it to the place_pose
        A simple sequence of pick and place motion sequences
        """
        movement_status = self.execute_action_pick(target_pose)
        if movement_status == MOVEMENT_STATUS_SUCCESS:
            if not place_pose:
                place_pose = target_pose
            movement_status = self.execute_action_place(place_pose)
            if movement_status == MOVEMENT_STATUS_SUCCESS:
                rospy.loginfo("Place action executed successfully")
                return MOVEMENT_STATUS_SUCCESS
            else:
                rospy.logerr("Placing phase during move action failed.")
                return MOVEMENT_STATUS_FAILED
        else:
            rospy.logerr("Picking phase during move action failed.")
            return MOVEMENT_STATUS_FAILED   

    def execute_action(self, target_pose=None, place_pose=None):
        """
        Main method for executing the action based on its type
        """
        # Pick action sequence
        if self.action_type == "pick" and self.movement_status == MOVEMENT_STATUS_WAITING:
            movement_execution_status = self.execute_action_pick(target_pose)
            return movement_execution_status
        
        if self.action_type == "place" and self.movement_status == MOVEMENT_STATUS_WAITING:
            movement_execution_status = self.execute_action_place(place_pose)
            return movement_execution_status
        
        if self.action_type == "move" and self.movement_status == MOVEMENT_STATUS_WAITING:
            movement_execution_status = self.execute_action_move(target_pose, place_pose)
            return movement_execution_status

        elif self.movement_status != MOVEMENT_STATUS_WAITING:
            rospy.logwarn("Movement in progress!")
            return MOVEMENT_STATUS_MOVING
        
        else:
            rospy.logwarn("Unknown action!")
            return MOVEMENT_STATUS_FAILED

def select_grasp_vector(grasps, option=0, threshold=-0.9):
    """
    Return pre-grasp vector based on selected option
    Inputs:
        graps: List of vectors (7,1)
        option: int, default = 0
    Output:
        selected_grasp: vector (7,1) or None if none found
    """
    f_vector = None # vector in option direction (forward ~ f)
    b_vector = None # vector against option direction (backward ~ b)
    if option == 0:
        # top-down pre-grasp
        f_vector = [0, 0, -1]
        b_vector = [0, 0, 1]
    elif option == 1:
        # right-left pre-grasp
        f_vector = [0, 1, 0]
        b_vector = [0, -1, 0]
    else:
        f_vector = [0, 0, -1]
        b_vector = [0, 0, 1]
    
    possible_grasps = []
    for i,grasp in enumerate(grasps):
        rot = tf_trans.quaternion_matrix(grasp[3:])[:3, :3]
        vector = np.dot(rot,b_vector)
        alignment = np.dot(f_vector, vector)
        if alignment >= threshold:
            possible_grasps.append(grasp)

    if len(possible_grasps) > 0:
        return possible_grasps[0]
    else:
        return None

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('motion_execution_node', anonymous=True)

    # Initialize and run the command sender
    manager = InputManager()
    sender = CommandSender()

    # setup zmq feedback to reasoner with motion execution status
    context = zmq.Context()
    motion_execution_feedback_pub = context.socket(zmq.PUB)
    motion_execution_feedback_pub.bind("tcp://*:5559")

    # give time for socket to bind
    rospy.sleep(1)

    # bool variable indicating wether the robot is holding an object or not
    holding_object = False

    # gdrn test data definition
    gdrnet_dummy_data = [ 
        {
            "name": "013_apple_1",
            "confidence": 0.91,
            "position": [0.65, 0.4, 0.6],
            "orientation": [0, 0, 0, 1],
        },
        {
            "name": "011_banana_1",
            "confidence": 0.95,
            "position": [0.56, -0.17, 0.67],
            "orientation": [0, 0, 0, 1],
        }
    ]

    # language pick dummy data 
    hri_dummy_pick_data = {
        "action": ["pick"],
        "action_probs": [1.0],
        "objects": ["013_apple_1", "011_banana_1"],
        "object_probs": [0.94, 0.55]
    }

    dummy_place_pose = [0.528, -0.478, 0.965, -0.587, -0.361, 0.640, -0.340] # for testing
    dummy_place_orientation = [0.5, 0.5, -0.5, 0.5] # used to specify orientation for placing objects
    gripper_size = 0.25 # m

    while not rospy.is_shutdown():
        try:
            # Returns True if we received execution request data from reasoner, else waits
            if manager.wait_for_data():
                rospy.loginfo(f"Waiting over, input data received")
                pass
            else:
                print("interrupt")
                raise rospy.ROSInterruptException

            rospy.loginfo(f"Detected GDRN objects : {manager.gdrn_objects}")
            rospy.loginfo(f"Current HRICommand : {manager.hricommand}")

            # Select the desired action based on its probability
            act_idx = int(np.argmax(manager.hricommand["action_probs"]))
            best_action = manager.hricommand["action"][act_idx]
            rospy.loginfo(f"Determined action : {best_action}")

            # Create action template with the corresponding action type
            action = ActionTemplate(
                action_type=best_action,
                command_sender=sender,
                confirm_poses=False # XXX: uncomment to skip movement confirmation
            )

            # Pick the target object and / or place pose
            target_pose = None
            place_pose = None

            if best_action == "pick" or best_action == "move":
                # Pick target object based on its probability
                obj_idx = int(np.argmax(manager.hricommand["object_probs"]))
                obj_name = manager.hricommand["objects"][obj_idx]
                rospy.loginfo(f"Target object: {obj_name}")

                # Look up its 6‑D pose in GDRN list
                obj_data = next(
                    (o for o in manager.gdrn_objects if o["name"] == obj_name),
                    None
                )
                if obj_data is None:
                    rospy.logwarn("Target object not found in GDRN list - aborting.")
                    continue

                # Load and transform its grasp vectors
                grasps_obj = manager.load_graps(obj_name) # poses in obj frame

                # Hotfix for gripper not going through the table - GDRN issue workaround
                obj_data["position"][2] = max(obj_data["position"][2], DEFAULT_TABLE_HEIGHT)

                target_grasps = manager.transform_grasps(grasps_obj, obj_data)
                target_pose = select_grasp_vector(target_grasps)

                rospy.loginfo(f"Computed target pose: {target_pose}")

            if best_action == "place" or best_action == "move":
                # Pick the place pose based on its probability
                pose_idx = int(np.argmax(manager.hricommand["place_probs"]))
                place_pose = manager.hricommand["place_poses"][pose_idx]

                # Add the orientation to the pose, lift the gripper by its size up
                place_pose = place_pose + dummy_place_orientation  # TODO: for now, hot sure how to determine better rotation

                # hotfix for above the table for place pose - GDRN issue workaround
                place_pose[2] = max(place_pose[2], DEFAULT_TABLE_HEIGHT)
                place_pose[2] += gripper_size
                rospy.loginfo(f"Chosen place pose: {place_pose}")

            # The loading failed due to unknown action
            if target_pose == None and place_pose == None:
                rospy.logwarn(f"Unknown / unsupported action: {best_action}")
                continue

            # Execute the motion through the defined template
            result = action.execute_action(target_pose=target_pose, place_pose=place_pose)
            rospy.loginfo(f"Action execution result: {result}")

            # Keep track if the system is holding an object or not TODO: not used anywhere
            if   best_action == "pick"  and result == "success": holding_object = True
            elif best_action == "place" and result == "success": holding_object = False
            elif best_action == "move" and result == "success": holding_object = False

            # Send feedback to Reasoner about the movement status
            feedback = json.dumps({"status": result})
            motion_execution_feedback_pub.send_string(feedback)
            rospy.loginfo("Published feedback: " + feedback)

            # allow the manager to wait for the next command
            manager.data_loaded = False

        except rospy.ROSInterruptException:
            pass
        
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt end")
