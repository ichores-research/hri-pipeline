import rclpy
import json
import numpy as np
import re
import time
from threading import Lock
from collections import deque   # <-- add (needed by the helper)

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from gesture_msgs.msg import HRICommand # Download https://github.com/ichores-research/modality_merging to workspace
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

import threading
rossem = threading.Semaphore()

DEICTIC_SEQUENCE_TOPIC = '/teleop_gesture_toolbox/deictic_sentence'
LANGUAGE_TOPIC = '/modality/nlp'

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

    # geometry_msgs.msg.Point
    if isinstance(obj, Point):
        return {"x": obj.x, "y": obj.y, "z": obj.z}

    # python 'array' module
    try:
        import array
        if isinstance(obj, array.array):
            return list(obj)
    except ImportError:
        pass

    # Anything already JSON-friendly is returned as-is
    return obj

def _dict_to_point(d):
    """
    Helper for converting {'x':..,'y':..,'z':..} to geometry_msgs.msg.Point
    """
    return Point(x=d["x"], y=d["y"], z=d["z"])

class RosNode(Node):
    def __init__(self):
        super().__init__("gesture_language_fusion_node")
        self.last_time_livin = time.time()

    def spin_once(self, sem=True):
        if sem:
            with rossem:
                self.last_time_livin = time.time()
                rclpy.spin_once(self)
        else:
            self.last_time_livin = time.time()
            rclpy.spin_once(self)

class DeicticSentenceInput:
    """
    Class that listens on DEICTIC_SEQUENCE_TOPIC and stores *all*
    deictic solutions in a list until fetched with get_sentence().
    The message on that topic is expected to be the JSON string produced
    by export_solutions_to_HRICommand().
    """
    def __init__(self, node: Node):
        self.node = node
        self._lock = Lock()
        self._solutions = []          # list of dicts
        self._enabled = True
        self._received_data = False

        # Subscriber to deictic gesture sequence
        self.sub = self.node.create_subscription(
            HRICommand,
            DEICTIC_SEQUENCE_TOPIC,
            self._sentence_callback,
            10
        )

    def _sentence_callback(self, msg: HRICommand):
        """
        Callback for gesture sentence message
        """
        if not self._enabled:
            return
        self._enabled = False

        # Load the gesture sequence to list fromo the message
        try:
            json_str = msg.data[0] 
            solutions_list = json.loads(json_str)
        except (IndexError, json.JSONDecodeError) as e:
            self.node.get_logger().error(f"[DeicticSentenceInput] invalid message: {e}")
            self._enabled = True
            return

        # Re-create geometry_msgs/Point fields from dicts
        for sol in solutions_list:
            sol["target_object_position"] = _dict_to_point(sol["target_object_position"])
            sol["line_points"][0] = _dict_to_point(sol["line_points"][0])
            sol["line_points"][1] = _dict_to_point(sol["line_points"][1])

        # Save the loaded data to class variable
        with self._lock:
            self._solutions = solutions_list
            self._received_data = True
        self.node.get_logger().info(
            f"[DeicticSentenceInput] received {len(solutions_list)} gestures"
        )
        
    def has_sentence(self) -> bool:
        """
        Returns True if we received data
        """
        with self._lock:
            return self._received_data

    def get_sentence(self):
        """
        Returns the stored list of solutions and clears the buffer
        so the next message can be accepted.
        """
        with self._lock:
            sols = self._solutions
            # clean the previous gesture sentence
            self._solutions = [] 

            # re-enable reception
            self._received_data = False
            self._enabled = True          

        self.node.get_logger().info(
            f"[DeicticSentenceInput] delivering {len(sols)} gestures"
        )
        return sols

class LanguageInput:
    """
    Class handling reception of language data from NLP.  
    One ROS message may contain *one* or *many* simple-sentence dicts.
    Each message is treated as ONE logical command and kept together
    in the buffer as 'list[dict]'.
    """
    def __init__(self, node: Node):
        self.node = node
        self._lock = Lock()
        self._buf = []
        self._enabled = True

        # Subscriber of language data from NLP
        self.sub = self.node.create_subscription(
            HRICommand,
            LANGUAGE_TOPIC,
            self._language_callback,
            10
        )

    def _language_callback(self, msg: HRICommand):
        """
        Calback handling the reception of language data from NLP
        """
        if not self._enabled:
            return
        self._enabled = False

        # Load data to list
        try:
            raw = json.loads(msg.data[0])          
        except (IndexError, json.JSONDecodeError) as e:
            self.node.get_logger().error(f"[LanguageInput] invalid JSON: {e}")
            self._enabled = True
            return

        if isinstance(raw, dict):
            raw = [raw]
        elif not isinstance(raw, list):
            self.node.get_logger().error(
                "[LanguageInput] unsupported payload type "
                f"({type(raw).__name__}); expected dict or list"
            )
            self._enabled = True
            return

        # Parse language data to required format, each elementary command
        norm_cmd = []
        for sent in raw:
            norm_cmd.append({
                "action"       : sent.get("action", ""),
                "objects"      : [sent.get("target_object", ""),
                                  sent.get("target_object2", "")],
                "action_param" : sent.get("action_parameter", ""),
                "objects_param": [sent.get("target_object_color", ""),
                                  sent.get("target_object_color2", "")],
                "raw_text": sent.get("raw_text", ""), # original raw text for the command
                "orig_idx": sent.get("orig_idx", ""), # index of the sentence in the original text
            })

        # Store all the commands as one combined request 
        with self._lock:
            self._buf.append(norm_cmd)

        self.node.get_logger().info(
            f"[LanguageInput] received NLP command "
            f"({len(norm_cmd)} sentence{'s' if len(norm_cmd)!=1 else ''})"
        )

    def has_sentence(self) -> bool:
        """
        Returns True if it received data and can be processed
        """
        with self._lock:
            return len(self._buf) > 0

    def get_sentence(self):
        """
        Returns the oldest buffered command (a list[dict]) and
        re-enables reception.
        """
        with self._lock:
            cmd = self._buf.pop(0) if self._buf else None
            self._enabled = True

        self.node.get_logger().info(
            f"[LanguageInput] delivering {len(cmd)} sentences"
        )
        return cmd

class GestureLanguageMerger(RosNode):
    """
    Class handling the correct assignment of gestures to language commands
    Sends the merged modalities in order of their required execution
    """
    def __init__(self, step_period: float = 0.2):
        super(GestureLanguageMerger, self).__init__()
        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.lock = Lock()

        # Loop wait period
        self.step_period = step_period

        # Load data listeners 
        self.language_sub = LanguageInput(self)
        self.deictic_sentence_sub = DeicticSentenceInput(self)

        # subscriber for the feedback from the reasoner
        self.create_subscription(
            Bool,
            "reasoner/start_episode",
            self._start_episode_cb,
            10
        )

        # Bool signaling waiting for the new episode signal from reasoner
        self.waiting_for_reasoner = False

        # fusion result publisher
        self.pub = self.create_publisher(
            HRICommand,
            "/modality/gesture_language_fusion",
            qos_profile
        )

        # create timer for the main loop for it to function
        self.timer = self.create_timer(0.2, self.main_loop) # 2.0 speedup? higher frequency

    def _start_episode_cb(self, msg: Bool):
        """
        Callback from reasoner to start waiting for new data
        """
        if msg.data:
            with self.lock:
                self.waiting_for_reasoner = False
            self.get_logger().info("Reasoning finished, starting new episode...")

    def main_loop(self):
        """
        Main loop of the fusion module, here the gestures are assigned to the language commands
        Publishes the message containing elementary language commands enriched by the list of gestures

        The entire message published is a list[dict], where each dict has the following keys:
        - action: desired action type (pick, place, move)
        - objects: list of object names [target, ref]
        - action_param: action parameter (relation)
        - objects_param: list of object params [param_target, param_ref], now only colors
        - raw_text: original wording of the elementary command in the sentence
        - orig_idx: order of the command in the original sentence
        - gestures: list of assigned gestures to this command (same structure from geture_sentence_detector)
        - gesture_role: determined role of gesture ("object" or "place"), only for 'move' action w. 1 gesture
        """
        #time.sleep(self.step_period)

        # If the previous command is being executed, wait for its end
        with self.lock:
            if self.waiting_for_reasoner:
                self.get_logger().info("Waiting for reasoner...")
                return

        # receive new data from both sources until we have data from both
        if not (
            self.language_sub.has_sentence()
            and self.deictic_sentence_sub.has_sentence()
        ):
            self.get_logger().warn("Waiting for language and / or deictic data...")
            self.deictic_sentence_sub._enabled = True
            self.language_sub._enabled = True
            return

        # Load the language and gesture data from input helpers
        nlp_data = self.language_sub.get_sentence()
        deictic_batch = self.deictic_sentence_sub.get_sentence()

        self.get_logger().info(f"[Fusion] NLP sentences   : {len(nlp_data)}")
        self.get_logger().info(f"[Fusion] Deictic gestures: {len(deictic_batch)}")

        # Pairing gestures to language commands
        if len(nlp_data) == 1:
            # Only one elementary sentence, attach all gestures to it.
            nlp_data[0]["gestures"] = deictic_batch
            if nlp_data[0]["action"] == "move" and len(deictic_batch) == 1:
                # We have to decide to which part of the action the gesture belongs to
                gesture_role = self.classify_gesture_for_move(nlp_data[0])
                if not gesture_role:
                    self.get_logger().warn("Cannot decide on gesture role, aborting...")
                    return
                
                nlp_data[0]["gesture_role"] = gesture_role

            # if 'move' with 0 or 2 gestures, nothing ambiguous, no need to determine gesture role
            fused_payload = nlp_data
        else:
            # More than one command - greedy pairing
            # We need two views of the same sentences:
            # - 'exec_order'  : the original list (execution order)
            # - 'spoken_order': a copy sorted by the position in the original sentence
            exec_order   = nlp_data
            spoken_order = sorted(exec_order, key=lambda s: s.get("orig_idx", 0))

            # List of gestures in the detection order
            remaining = list(deictic_batch) 

            # Process the messages in spoken order, the assigned gestures are also in the exec_order
            for sentence in spoken_order:
                # Determine the number of needed gestures based on the raw text
                need = self.gestures_needed_number(sentence)
                self.get_logger().info(f"Sentence '{sentence['raw_text']}' - gestures: {need}")

                # If we need more gestures than we have, there is something wrong
                if len(remaining[:need]) != need:
                    self.get_logger().warn(f"[Fusion] Not enough gestures remaining!")
                    return

                sentence["gestures"] = remaining[:need]
                del remaining[:need]

                # For action 'move' w. 1 gesture determine its role
                if sentence["action"].lower() == "move" and len(sentence["gestures"]) == 1:
                    gesture_role = self.classify_gesture_for_move(sentence)
                    if not gesture_role:
                        self.get_logger().warn("Cannot decide on gesture role, aborting...")
                        return
                
                    sentence["gesture_role"] = gesture_role

            # If we have some unassigned gestures, there is something wrong
            if remaining:
                self.get_logger().warn(f"[Fusion] Unassigned gestures: {len(remaining)}")
                return

            # To the payload put the commands again in the execution order
            fused_payload = exec_order

        # Publish the message as json string in HRICommand message
        json_str = json.dumps(to_jsonable(fused_payload))

        msg = HRICommand(data=[json_str])
        self.pub.publish(msg)

        self.get_logger().info(
            f"[Fusion] published fusion message: {json_str}"
        )

        # Start waiting for the processing of the commands
        with self.lock:
            self.waiting_for_reasoner = True
        

    def classify_gesture_for_move(self, sentence_dict: dict):
        """
        Decide whether the gesture in a 'move' elementary command sentence is meant
        for the target *object* or the target *place*.

        Input:
            dictionary contaning the language data for the elementary command

        Returns
            "object" | "place" | None
        """
        # Get the action param
        rel = (sentence_dict.get("action_param") or "").lower().strip()

        # if no relation detected, something is wrong
        if not rel:
            self.get_logger().warn( f"[Fusion] Relation field empty")
            return None

        # if relation is place, the gesture is specifying place pose
        if rel == "here":
            return "place"

        # Normalize the text and split the sentence at the action parameter
        raw_text = (sentence_dict.get("raw_text") or "").lower()
        raw_text = re.sub(r'[^\w\s]', ' ', raw_text) # strip punctuation

        before_rel, _, after_rel = raw_text.partition(rel)

        # if action param not in original command, something is wrong
        if _ == "":
            self.get_logger().warn( f"[Fusion] Relation field not in original sentence")
            return None

        # If there is a demonstrative pronoun before the relation, the gesture specifies target object
        if re.search(r'\b(this|that)\b', before_rel):
            return "object"

        # If there is a demonstrative pronoun after the relation, the gesture specifies target place
        if re.search(r'\b(this|that)\b', after_rel):
            return "place"

        # Otherwise the meaning is unclear, role cannot be determined, end the command processing
        self.get_logger().warn( f"[Fusion] Meaning of the relation is unclear")
        return None

    def gestures_needed_number(self, fused_dict: dict) -> int:
        """
        Determines the number of required gestures from the language
        based on the number of demonstrative pronouns and "here"

        Input:
            sentence: dict of 1 fused elementary command
        
        Returns:
            number of gestures to be assigned to the command
        """
        act = fused_dict["action"].lower()
        param = fused_dict["action_param"].lower()
        txt = fused_dict["raw_text"].lower()

        has_this_that = bool(re.search(r'\b(this|that)\b', txt))

        # For 'pick' action only the demonstrative pronoun matters
        if act == "pick":
            return 1 if has_this_that else 0

        # For 'place' action we need gesture for action param 'here' or demonstrative pronoun
        if act == "place":
            if param == "here":
                return 1
            
            return 1 if has_this_that else 0

        # For 'move' action we have several options
        if act == "move":
            if param == "here":
                # 1 gesture for 'here'
                # try to find demonstrative pronoun in the first part of the sentence
                before, _, _ = txt.partition("here")
                obj_needs_gesture = bool(re.search(r'\b(this|that)\b', before))
                return 2 if obj_needs_gesture else 1

            # if param is not 'here' just count the demonstrative pronouns, max 2 
            n_demonstratives = len(re.findall(r'\b(this|that)\b', txt))
            return min(n_demonstratives, 2)

        # Otherwise we got unsupported action
        self.get_logger().warn(f"Unsuported action '{act}'")
        return 0

def main():
    rclpy.init()
    print("running init...")
    merger = GestureLanguageMerger()
    merger.get_logger().info("GestureLanguageMerger initialized.")
    try:
        rclpy.spin(merger)
    except KeyboardInterrupt:
        print("Ending by User Interrupt")
    finally:
        merger.destroy_node()

if __name__ == '__main__':
    main()
