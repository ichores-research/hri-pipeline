#! /usr/bin/env python3
import rospy
from object_detector_msgs.srv import detectron2_service_server, estimate_pointing_gesture, estimate_poses
from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorResult, GenericImgProcAnnotatorFeedback, GenericImgProcAnnotatorGoal
import actionlib
from sensor_msgs.msg import Image, RegionOfInterest

import tf
import tf.transformations as tf_trans
import numpy as np
import open3d as o3d
import yaml
import os
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion

from utils import *

class PoseCalculator:
    def __init__(self):
        self.image_publisher = rospy.Publisher('/pose_estimator/image_with_roi', Image, queue_size=10)
        self.bridge = CvBridge()

        self.models = load_models("/root/task/datasets/ycb_ichores/models", "/root/config/ycb_ichores.yaml")
        self.grasp_annotations = load_grasp_annotations("/root/task/datasets/ycb_ichores/grasp_annotations", "/root/config/ycb_ichores.yaml")

        # self.client = actionlib.SimpleActionClient('/pose_estimator/gdrnet', 
        #                                            GenericImgProcAnnotatorAction)
        #self.client.wait_for_server()

        # self.server = actionlib.SimpleActionServer('/pose_estimator',
        #                                            GenericImgProcAnnotatorAction,
        #                                            execute_cb=self.get_poses_robokudo,
        #                                            auto_start=False)

        self.color_frame_id = rospy.get_param('/pose_estimator/color_frame_id')
        self.grasp_frame_id = rospy.get_param('/pose_estimator/grasp_frame_id')

        self.marker_id = 0

        #self.server.start()

    def detect_objects(self, rgb):
        rospy.wait_for_service('detect_objects')
        try:
            detect_objects_service = rospy.ServiceProxy('detect_objects', detectron2_service_server)
            response = detect_objects_service(rgb)
            return response.detections.detections
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def estimate_object_poses(self, rgb, depth, detection):
        rospy.wait_for_service('estimate_poses')
        try:
            estimate_poses_service = rospy.ServiceProxy('estimate_poses', estimate_poses)
            response = estimate_poses_service(detection, rgb, depth)
            return response.poses
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def detect_pointing_gesture(self, rgb, depth):
        rospy.wait_for_service('detect_pointing_gesture')
        try:
            detect_pointing_gesture_service = rospy.ServiceProxy('detect_pointing_gesture', estimate_pointing_gesture)
            response = detect_pointing_gesture_service(rgb, depth)
            return response
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def get_grasps(self, obj_poses):
        grasps = []
        for idx, obj_pose in enumerate(obj_poses):
            obj_name = obj_pose.name
            grasps_obj_frame = self.grasp_annotations.get(obj_name, None)['grasps']
            grasps_world_frame = transform_grasp_obj2world(grasps_obj_frame, obj_pose.pose)
            grasps.append(grasps_world_frame)

        return grasps
    
    # def estimate_poses(self, rgb, depth, detections):
    #     try:
    #         goal = GenericImgProcAnnotatorGoal()
    #         goal.rgb = rgb
    #         goal.depth = depth
    #         bb_detections = []
    #         class_names = []
    #         description = ''
    #         ind_detections = np.arange(0, len(detections), 1)
    #         for detection, index in zip(detections, ind_detections):
    #             roi = RegionOfInterest()
    #             roi.x_offset = detection.bbox.ymin
    #             roi.y_offset = detection.bbox.xmin
    #             roi.height = detection.bbox.xmax - detection.bbox.xmin
    #             roi.width = detection.bbox.ymax - detection.bbox.ymin
    #             roi.do_rectify = False

    #             bb_detections.append(roi)
    #             class_names.append(detection.name)
    #             if index == 0:
    #                 description = description + f'"{detection.name}": "{detection.score}"'
    #             else:
    #                 description = description + f', "{detection.name}": "{detection.score}"'
    #         description = '{' + description + '}'
    #         goal.bb_detections = bb_detections
    #         goal.class_names = class_names
    #         goal.description = description
    #         self.client.send_goal(goal)
    #         self.client.wait_for_result()
    #         result = self.client.get_result()
    #     except rospy.ServiceException as e:
    #         print("Service call failed: %s" % e)

    #     return result

    def publish_annotated_image(self, rgb, detections):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        for detection in detections:
            xmin = int(detection.bbox.ymin)
            ymin = int(detection.bbox.xmin)
            xmax = int(detection.bbox.ymax)
            ymax = int(detection.bbox.xmax)

            font_size = 1.0
            line_size = 3

            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), line_size)

            class_name = detection.name
            score = detection.score
            label = f"{class_name}: {score:.2f}"
            cv2.putText(cv_image, label, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), line_size)

        # Publish annotated image
        annotated_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_publisher.publish(annotated_image_msg)

        # Display image for debugging
        # cv2.imshow("Annotated Image", cv_image)
        # cv2.waitKey(10)

    # def get_poses_robokudo(self, goal):
    #     res = GenericImgProcAnnotatorResult()
    #     res.success = False
    #     res.result_feedback = "calculated: "
    #     feedback = GenericImgProcAnnotatorFeedback()

    #     # === check if we have an image ===
    #     if goal.rgb is None or goal.depth is None:
    #         print("no images available")
    #         res.result_feedback = "no images available"
    #         res.success = False
    #         self.server.set_succeeded(res)
    #         return
    #     rgb = goal.rgb
    #     depth = goal.depth
    #     print('Perform detection with YOLOv8 ...')
    #     detections = self.detect_objects(rgb)
    #     print("... received detection.")

    #     if detections is None or len(detections) == 0:
    #         print("nothing detected")
    #         self.server.set_aborted(res)
    #         return
    #     else:
    #         print('Perform pose estimation with GDR-Net++ ...')
    #         try:
    #             res = self.estimate_poses(rgb, depth, detections)

    #         except Exception as e:
    #             rospy.logerr(f"{e=}")

    #     # res.class_names = class_names
    #     res.result_feedback = res.result_feedback + ", class_names"
    #     # res.class_confidences = class_confidences
    #     res.result_feedback = res.result_feedback + ", class_confidences"
    #     # res.pose_results = pose_results
    #     res.result_feedback = res.result_feedback + ", pose_results"
    #     res.success = True
    #     print(f"{res=}")
    #     self.server.set_succeeded(res)

    def publish_mesh_marker(self, cls_name, quat, t_est):
        vis_pub = rospy.Publisher("/gdrnet_meshes_estimated", Marker, latch=True)
        model_data = self.models.get(cls_name, None)
        model_vertices = np.array(model_data['vertices'])/1000

        marker = Marker()
        marker.header.frame_id = self.grasp_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.TRIANGLE_LIST
        marker.ns = cls_name
        marker.action = Marker.ADD
        marker.pose.position.x = t_est[0]
        marker.pose.position.y = t_est[1]
        marker.pose.position.z = t_est[2]

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        shape_vertices = 3*int((model_vertices.shape[0] - 1)/3)
        for i in range(shape_vertices):
            pt = Point(x = model_vertices[i, 0], y = model_vertices[i, 1], z = model_vertices[i, 2])
            marker.points.append(pt)
            rgb = ColorRGBA(r = 1, g = 0, b = 0, a = 1.0)
            marker.colors.append(rgb)

        vis_pub.publish(marker)

    def publish_grasp_marker(self, transformed_grasps):
        marker_pub = rospy.Publisher("/gdrnet_grasps_vis", MarkerArray, latch=True)

        marker_array = MarkerArray()
        align_x_to_z = tf_trans.quaternion_from_euler(0, np.pi / 2, 0)

        for idx, grasp_matrix in enumerate(transformed_grasps):
            grasp_matrix = np.array(grasp_matrix).reshape(4, 4)

            position = grasp_matrix[:3, 3]
            orientation_quat = tf_trans.quaternion_from_matrix(grasp_matrix)
            adjusted_orientation = tf_trans.quaternion_multiply(orientation_quat, align_x_to_z)

            marker = Marker()
            marker.header.frame_id = self.color_frame_id        # this is only for visualization!
            marker.header.stamp = rospy.Time.now()
            marker.ns = "grasp_arrows"
            marker.id = idx

            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.x = adjusted_orientation[0]
            marker.pose.orientation.y = adjusted_orientation[1]
            marker.pose.orientation.z = adjusted_orientation[2]
            marker.pose.orientation.w = adjusted_orientation[3]

            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        marker_pub.publish(marker_array)

if __name__ == "__main__":
    rospy.init_node("calculate_poses")
    try:
        pose_calculator = PoseCalculator()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rgb = rospy.wait_for_message(rospy.get_param('/pose_estimator/color_topic'), Image)
            depth = rospy.wait_for_message(rospy.get_param('/pose_estimator/depth_topic'), Image)

            t0 = time.time()
            detections = pose_calculator.detect_objects(rgb)
            time_detections = time.time() - t0

            if detections is not None:
                pose_calculator.publish_annotated_image(rgb, detections)
                for detection in detections:
                    print(detection.name)

            print()
            print("... received object detection.")

            print('Perform Pointing Detection...')
            t0 = time.time()
            joint_positions = pose_calculator.detect_pointing_gesture(rgb, depth)
            time_pointing = time.time() - t0
            print('... received pointing gesture.')

            estimated_poses = []
            estimated_grasps = None

            t0 = time.time()
            if detections is None or len(detections) == 0:
                print("nothing detected")
            else:
                try:
                    for detection in detections:
                        if not detection.name == "036_wood_block":
                            estimated_pose = pose_calculator.estimate_object_poses(rgb, depth, detection)[0]
                            estimated_poses.append(estimated_pose)
                    print(estimated_poses)
                    estimated_grasps = pose_calculator.get_grasps(estimated_poses)    
                except Exception as e:
                    rospy.logerr(f"{e}")

            time_object_poses = time.time() - t0

            # show grasps in RViz
            if estimated_grasps is not None:
                for grasp in estimated_grasps:
                    pose_calculator.publish_grasp_marker(grasp)

            # # New step: Check which object the human is pointing to
            # t0 = time.time()
            # if estimated_poses and joint_positions is not None:
            #     elbow = joint_positions.elbow
            #     wrist = joint_positions.wrist
            #     min_distance = float('inf')
            #     pointed_object = None
            #     threshold = 0.3  # 0.5 meters              

            #     pointed_object_pose = None
            #     for idx, pose_result in enumerate(estimated_poses.pose_results):
            #         object_position = pose_result.position
            #         distance = calculate_distance_to_line(object_position, elbow, wrist)
            #         if distance < min_distance:
            #             min_distance = distance
            #             pointed_object = estimated_poses.class_names[idx]
            #             pointed_object_pose = pose_result

            #     if min_distance < threshold:
            #         R = np.array([pointed_object_pose.orientation.x, pointed_object_pose.orientation.y,  pointed_object_pose.orientation.z, pointed_object_pose.orientation.w])
            #         t = np.array([pointed_object_pose.position.x, pointed_object_pose.position.y, pointed_object_pose.position.z])
            #         pose_calculator.publish_mesh_marker(pointed_object, R, t)
            #         print(f"The human is pointing to the object: {pointed_object}")
            #         print()

            # time_point_checker = time.time() - t0
            # # Print the timed periods
            # print(f"Time for object detection: {time_detections:.2f} seconds")
            # print(f"Time for pointing detection: {time_pointing:.2f} seconds")
            # print(f"Time for object pose estimation: {time_object_poses:.2f} seconds")
            # print(f"Time for pointing checker: {time_point_checker:.2f} seconds")
            # print()
            #rate.sleep()

    except rospy.ROSInterruptException:
        pass

