#!/usr/bin/env python
import numpy as np
import threading, json, os, yaml
import tf.transformations as tf_trans

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String

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


class InputManager():
    def __init__(self):
        # print(os.getcwd())
        self.grasp_annotations = self.load_grasp_annotations('src/tiago_gal_control/src/annotations_tiago', 'src/tiago_gal_control/src/ycb_ichores.yaml')
        self.load_gdrnet()
        pass
    
    def load_gdrnet(self):
        msg = rospy.wait_for_message('/gdrnet_object_poses', String, timeout=1)
        detected_names = {}
        self.gdrn_objects = json.loads(msg.data)
        self.gdrn_names = []
        for obj in self.gdrn_objects:
            obj_name = obj['name']
            if obj_name in detected_names.keys():
                detected_names[obj_name] += 1
                obj['name'] = "%s_%d" % (obj_name, detected_names[obj_name])
                self.gdrn_names.append(obj['name'])
            else:
                detected_names[obj_name] = 1
                obj['name'] = "%s_%d" % (obj_name, detected_names[obj_name])
                self.gdrn_names.append(obj['name'])

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
                # print(grasps.shape)
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


class ActionTemplate():
    def __init__(self, action_type, command_sender):
        self.action_type = action_type
        self.command_sender = command_sender
        self.movement_status = MOVEMENT_STATUS_WAITING

        self.lock = threading.Lock()
        self.bridge_logs_thread = threading.Thread(target=self.bridge_logs_listener)
        self.bridge_logs_thread.start()

    def bridge_logs_listener(self):
        rospy.Subscriber("/bridge_logs", String, self.bridge_logs_callback)

    def bridge_logs_callback(self, msg):
        """
        Callback function for the logs from bridge
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
        Function for getting the current movemement status
        """
        self.lock.acquire()
        movement_status = self.movement_status
        self.lock.release()
        return movement_status
    
    def set_movement_status(self, status):
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
            rospy.signal_shutdown('fail')
            return
        rospy.loginfo("Movement successful!")

    def execute_action(self, target_pose):
        """
        Main method for executing the action based on its type
        """
        # Pick action sequence
        target_x = target_pose[0]
        target_y = target_pose[1]
        target_z = target_pose[2]
        target_qx = target_pose[3]
        target_qy = target_pose[4]
        target_qz = target_pose[5]
        target_qw = target_pose[6]

        if self.action_type == "pick" and self.movement_status == MOVEMENT_STATUS_WAITING:
            # Step 1: Open gripper to be ready for picking an object
            # Based on the target pose, the arm is simply chosen according to the y-axis value
            if target_pose[1] <= 0:
                gripper_command = str(GRIPPER_OPEN_RIGHT)
            else:
                gripper_command = str(GRIPPER_OPEN_LEFT)
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_gripper_command(gripper_command)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 1 finished")

            # Step 2: Put the arm high to avoid collision with table
            pose = [0.2, -0.7, 1.0, 0, 0.707, 0.0] #TODO: Find suitable high position
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_pose("any", pose)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 2 finished")

            # Step 3: Move above the object ~ pre-grasp vector
            pose = target_pose #TODO: Find suitable hand rotation
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_pose("any", pose)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 3 finished")
            
            confirmation = input("Is it safe to continue from this pre-grasp position? y/n: ")
            if not (confirmation == 'y' or confirmation == 'yes'):
                return MOVEMENT_STATUS_CANCELED
            
            # Step 4: move to the object position (grasp pose)
            pose = [target_x, target_y, target_z-0.08, target_qx, target_qy, target_qz, target_qw] #TODO: Find suitable hand rotation
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_pose("any", pose)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 4 finished")

            # Step 5: Close the gripper
            gripper_command = GRIPPER_BOTH_CLOSED
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_gripper_command(gripper_command)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 5 finished")

            # Step 6: ad Step 3
            pose = [target_x, target_y, target_z, target_qx, target_qy, target_qz, target_qw] #TODO: Find suitable hand rotation
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_pose("any", pose)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 6 finished")

            # Step 7: ad Step 2
            pose = [0.2, -0.7, 1.0, 0.0, 0.707, 0] #TODO: Find suitable high position
            self.set_movement_status(MOVEMENT_STATUS_MOVING)
            self.command_sender.send_pose("any", pose)
            self.wait_for_movement_finish()
            rospy.loginfo("Step 7 finished")

            self.set_movement_status(MOVEMENT_STATUS_WAITING)

            return MOVEMENT_STATUS_SUCCESS

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

    try:
        # Initialize the node
        rospy.init_node('command_sender_node', anonymous=True)
        # Initialize and run the command sender
        sender = CommandSender()
        manager = InputManager()

        gdrnet_dummy_data = [ 
            {
                "name": "apple",
                "confidence": 0.91,
                "position": [0.65, 0.4, 0.6],
                "orientation": [0, 0, 0, 1],
            },
            {
                "name": "banana",
                "confidence": 0.95,
                "position": [0.56, -0.17, 0.67],
                "orientation": [0, 0, 0, 1],
            }
        ]

        hri_dummy_data = {
            "actions": ["pick"],
            "action_probs": [1.0],
            "objects": manager.gdrn_names,
            "objects_probs": [0.94, 0.55]
        }

        # Determine the best action
        best_action_index = np.argmax(hri_dummy_data["action_probs"])
        best_action = hri_dummy_data["actions"][best_action_index]
        rospy.loginfo("Determined action: %s" % best_action)

        action = ActionTemplate(action_type=best_action, command_sender=sender)

        # Determine the target object
        target_object_index = np.argmax(hri_dummy_data["objects_probs"])
        target_object = hri_dummy_data["objects"][target_object_index]
        rospy.loginfo("target_object: %s" % target_object)

        # Get the target object position
        target_pose = None
        for obj in manager.gdrn_objects:
            if obj["name"] == target_object:
                grasps_obj_frame = manager.load_graps(target_object)
                target_grasps = manager.transform_grasps(grasps_obj_frame, obj)
                rospy.loginfo(f"Loaded pre-grasp vectors, first one:\n{target_grasps[0]}")
                target_pose = select_grasp_vector(target_grasps)
                break
        if not target_pose:
            rospy.logwarn("Object not detected by the GDRNet")
            exit(100)

        action_result = action.execute_action(target_pose)

        rospy.loginfo("Action execution result: %s" % action_result)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
