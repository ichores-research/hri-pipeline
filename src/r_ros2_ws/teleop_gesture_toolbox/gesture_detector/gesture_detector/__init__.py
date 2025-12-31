
import os
path = os.path.dirname(os.path.abspath(__file__))
saved_models_path = path+"/saved_models/"
gesture_data_path = os.path.expanduser("~/ichores_ws/build/gesture_detector/gesture_detector/gesture_data")
"~/teleop_2_ws/src/teleop_gesture_toolbox/gesture_detector/gesture_detector/gesture_data/" #path+"/gesture_data/"

package_path = "/".join(path.split("/")[:-1])
