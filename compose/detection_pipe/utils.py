import yaml
import os
import numpy as np
import open3d as o3d
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker, MarkerArray
import rospy

def load_grasp_annotations(folder_path, yaml_file_path):
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    
    names = yaml_data.get('names', {})
    grasp_annotations = {}

    for obj_id, obj_name in names.items():
        filename = f"obj_{int(obj_id):06d}.npy"
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            grasps = np.load(file_path)
            print(grasps.shape)
            grasp_annotations[obj_name] = {'grasps': grasps}
    return grasp_annotations

def load_models(folder_path, yaml_file_path):
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    
    names = yaml_data.get('names', {})
    models = {}

    for obj_id, obj_name in names.items():
        filename = f"obj_{int(obj_id):06d}.ply"
        model_path = os.path.join(folder_path, filename)
        if os.path.exists(model_path):
            model = o3d.io.read_point_cloud(model_path)
            vertices = np.asarray(model.points)
            colors = np.asarray(model.colors) if model.colors else None
            models[obj_name] = {'vertices': vertices, 'colors': colors}
    return models

def transform_grasp_obj2world(grasps, pose):
    transformed_grasps = []

    # Convert object quaternion to a 4x4 transformation matrix, then extract the 3x3 rotation matrix
    obj_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    obj_transform = tf_trans.quaternion_matrix(obj_quat)  # This gives a 4x4 matrix
    obj_R = obj_transform[:3, :3]  # Extract the 3x3 rotation part
    obj_t = np.array([pose.position.x, pose.position.y, pose.position.z])  # Translation vector

    for grasp in grasps:
        # Convert the grasp to a 4x4 matrix
        grasp_matrix = np.array(grasp).reshape(4, 4)

        # Apply rotation and translation to transform the grasp to world coordinates
        transformed_grasp_matrix = np.eye(4)
        transformed_grasp_matrix[:3, :3] = np.dot(obj_R, grasp_matrix[:3, :3])  # Rotate
        transformed_grasp_matrix[:3, 3] = np.dot(obj_R, grasp_matrix[:3, 3]) + obj_t  # Rotate and translate

        # Flatten the transformed matrix and store it
        transformed_grasps.append(transformed_grasp_matrix.flatten())

    return np.array(transformed_grasps)

def calculate_distance_to_line(point, line_start, line_end):
    """
    Calculate the normal distance from a point to a line defined by two points.
    """
    line_vec = np.array([line_end.x - line_start.x, line_end.y - line_start.y, line_end.z - line_start.z])
    point_vec = np.array([point.x - line_start.x, point.y - line_start.y, point.z - line_start.z])
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    nearest = t * line_unitvec
    distance = np.linalg.norm(nearest - point_vec_scaled) * line_len
    return distance