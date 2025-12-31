#!/usr/bin/env python
import time, rclpy, json
import numpy as np
import threading
import rclpy.logging

from threading import Lock
from collections import deque

from scene_getter.scene_getting import SceneGetter
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from gesture_detector.hand_processing.hand_listener import HandListener
from gesture_detector.utils.utils import CustomDeque

from gesture_sentence_maker.segmentation_task.deictic_solutions_plot import deictic_solutions_plot_save
from gesture_sentence_maker.segmentation_task.deictic_segment import find_pointed_objects_timewindowmax

from gesture_msgs.msg import HRICommand
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from rclpy.node import Node

rossem = threading.Semaphore()

def to_jsonable(obj):
    """
    Helper for converting obj (possibly nested) into JSON-serialisable primitives
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, deque)):
        return [to_jsonable(v) for v in obj]

    # NumPy scalars & arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    # python 'array' module
    try:
        import array
        if isinstance(obj, array.array):
            return list(obj)
    except ImportError:
        pass

    # geometry_msgs.msg.Point
    if isinstance(obj, Point):
        return {"x": obj.x, "y": obj.y, "z": obj.z}

    # anything already JSON-friendly just falls through
    return obj

class RosNode(Node):
    def __init__(self):
        super().__init__("gesture_deictic_sentence_node")
        self.last_time_livin = time.time()

    def spin_once(self, sem=True):
        if sem:
            with rossem:
                self.last_time_livin = time.time()
                rclpy.spin_once(self)
        else:
            self.last_time_livin = time.time()
            rclpy.spin_once(self)

class GestureDeicticSentence(PointingObjectGetter, SceneGetter, HandListener, RosNode):
    def __init__(self, step_period: float = 0.2):
        super(GestureDeicticSentence, self).__init__()

        self.gesture_sentence_publisher = self.create_publisher(HRICommand, "/teleop_gesture_toolbox/deictic_sentence", 5)

        # sentence data
        self.deictic_solutions = CustomDeque()

        # variables for velocity / gesture detection
        self.last_line_points = None
        self.last_gesture_timestamp = None
        self.last_frame_seq = None
        self.steady_start_sec = None
        self.gesture_detected = False

        self.lock = Lock()
        
        self.step_period = step_period # main loop sleep period

        # Flags used in feedback from other modules
        self.continue_episode = self._continue_episode
        self.episode_end_flag = False
        self.wait_for_reasoner = self._waiting_for_reasoner
        self.reasoner_wait_flag = False

        # deictic sentence detection constants 
        self.MOVEMENT_SPEED_THRESHOLD = 0.1
        self.GESTURE_DWELL_TIME_SEC = 1.0 # s

        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # subscriber to nlp processed output
        self.create_subscription(
            HRICommand,
            "/modality/nlp",
            self.signal_episode_end,
            qos_profile
        )

        # subscriber to the reasoner feedback
        self.create_subscription(
            Bool,
            "reasoner/start_episode",
            self.start_episode_cb,
            10
        )

    def signal_episode_end(self, msg):
        """
        Callback from nlp that the language is already processed
        """
        # s = msg.data[0]
        # s = s.replace("'", '"')
        # self.nlp_msg = json.loads(s)
        # self.nlp_target_objects = msg_dict['target_object']
        #self.nlp_target_objects_stamp = msg_dict['target_obect_stamp']
        #self.get_logger().info(f" Received NLP: {msg_dict}")
        self.get_logger().info(f"Received NLP, end of episode.")
        self.episode_end_flag = True
    
    def start_episode_cb(self, msg: Bool):
        """
        Callback from reasoner that we should start a new gesture episode
        """
        if msg.data:
            with self.lock:
                self.reasoner_wait_flag = False
                self.episode_end_flag = False
            self.get_logger().info("Reasoning finished, starting new episode...")
        
    def _continue_episode(self):
        """
        Returns True if we should end the episode
        """
        with self.lock:
            return not self.episode_end_flag
    
    def _waiting_for_reasoner(self):
        """
        Returns True if we are waiting for the result of the previous gesture sequence
        """
        with self.lock:
            return self.reasoner_wait_flag

    def step(self): # speedup? place for petr vanc code?
        """
        Performs one pointing gesture solution evaluation
        1. Waits step_period
        2. Gets the newest hand frame from external camera
        3. Computes hand velocity & time for which the hand is still 
        4. If still long enough, treat it as desired gesture and store it
        """
        # sleep for determined duration
        time.sleep(self.step_period)

        # if we did not receive any nlp data, continue with episode
        if self.continue_episode():
            # There has to be some object in the scene
            if not self.target_object_valid():
                self.get_logger().info("Non-valid target object")
                return
            
            # Load a deictic solution
            solution = self.get_target_object()

            # Check if hand in the view and we have a new hand frame
            if not self.hand_frames or self.hand_frames[-1].seq == self.last_frame_seq: 
                self.get_logger().info("Hand not detected.")
                return

            self.last_frame_seq = self.hand_frames[-1].seq

            now = time.time()

            # Compute cumulative pose change of line points
            
            pose_change = 0
            points = solution["line_points"]

            if self.last_line_points is None:
                velocity = 0.0
            else:
                last_points = self.last_line_points
                for index in range(len(points)):
                    diff = np.sqrt(
                        ((points[index].x - last_points[index].x)) ** 2 +
                        ((points[index].y - last_points[index].y)) ** 2 +
                        ((points[index].z - last_points[index].z)) ** 2
                    )
                    pose_change += diff
                velocity = pose_change

            # For now, it is pose change, not velocity
            solution["hand_velocity"] = velocity 
            self.get_logger().info(f"hand_velocity: {velocity:.3f} m")

            # dwell-time gesture detection
            # reset dwell timer if hand is moving
            if velocity > self.MOVEMENT_SPEED_THRESHOLD:
                self.steady_start_sec = None
                self.gesture_detected = False

            else:
                # Hand movement below threshold, measure for how long
                if self.steady_start_sec is None:
                    self.steady_start_sec = now
                elif (now - self.steady_start_sec) >= self.GESTURE_DWELL_TIME_SEC:
                    # If hand stready for long enough, save the deictic solution as a pointing gesture
                    # Do not save another one until after the hand moves and stops again 
                    if not self.gesture_detected:
                        self.get_logger().info(f"Pointing gesture detected, pointed object: {solution['target_object_name']}")
                        self.gesture_detected = True
                        self.deictic_solutions.append(solution)

            # Save line points for next iteration
            self.last_line_points = points
            self.last_gesture_timestamp = now

        # if we did receive nlp data, end current episode and send the gesture sequence
        elif not self.continue_episode() and not self.wait_for_reasoner():
            self._episode_end_cleanup()

        # Otherwise wait for feedback from reasoner that the previous commands were processed
        else: 
            self.get_logger().info("Waiting for reasoner...")
            return

    def _episode_end_cleanup(self):
        """
        Function publishing the detected gesture sequence for fusion with language.
        """
        # signal end of episode
        with self.lock:
            self.episode_end_flag = True
        
        # publish the gestures in current episode and clear the buffer for the next one 
        time.sleep(0.5) # speedup?
        self.gesture_sentence_publisher.publish(
            self.export_solutions_to_HRICommand(self.deictic_solutions)
        )
        self.deictic_solutions = CustomDeque()

        # Wait until the command is processes and executed
        with self.lock:
            self.reasoner_wait_flag = True
    
    def export_solutions_to_HRICommand(self, solutions_deque: CustomDeque):
        """
        Wrap all solutions into a JSON string inside HRICommand.data[0].
        """
        # First convert deque to list
        jsonable = to_jsonable(list(solutions_deque)) 
        #self.get_logger().info(f"NLP: {self.nlp_msg}")
        self.get_logger().info(f"Gestures: {jsonable}")
        
        # Fold the data in HRICommand message
        json_str = json.dumps(jsonable)
        self.get_logger().info(f"Publishing {len(self.deictic_solutions)} gestures: {json_str}")
        return HRICommand(data=[json_str])


def spinning_threadfn(sp):
    while rclpy.ok():
        sp.spin_once(sem=True)
        time.sleep(0.01) # speedup?

def main():
    rclpy.init()
    print("running init...")
    sentence_processor = GestureDeicticSentence()
    spinning_thread = threading.Thread(target=spinning_threadfn, args=(sentence_processor, ), daemon=True)
    spinning_thread.start()
    
    sentence_processor.get_logger().info("Detecting gestures...")
    while rclpy.ok():
        sentence_processor.step()
        

if __name__ == "__main__":
    main()
