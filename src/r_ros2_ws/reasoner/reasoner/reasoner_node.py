#!/usr/bin/python3

import json
import numpy as np
import rclpy
import re
import threading
import time
import zmq

from rclpy.node import Node

from gesture_msgs.msg import HRICommand
from std_msgs.msg import Header, Bool
from reasoner.classes import (
    FusedSentenceInput,
    GDRNetInput,
    ReasonerTester,
) 

class Reasoner(Node):
    """
    Class responsible for merging the informaion from camera,
    language and gestures to determine the target objects
    and / or place poses for the sequence of elementary commands
    """
    # Action type to handler definition
    ACTION_HANDLERS = {
        "pick":  "_handle_pick",
        "move":  "_handle_move", 
        "place": "_handle_place",
        # XXX: add new handlers for new action types here
    }

    # Definitions of relations and roles allowed
    PLACE_ALLOWED_RELATIONS = [
        "left", "right", "front", "behind", "color", "shape", "here"
    ] 
    PICK_ALLOWED_RELATIONS = [
        "left", "right", "front", "behind", "color", "shape",
    ]
    PLACE_SUPPORTED_SPATIAL_RELATIONS = {"left", "right", "front", "behind"}
    MOVE_SUPPORTED_GESTURE_ROLES = {"object", "place"}

    # Constants used in reasoning
    FALLBACK_TABLE_HEIGHT = 0.50 # m XXX: CHANGE ACCORDING TO YOUR TABLE HEIGHT
    REFERENCE_POSE_OFFSET = 0.12 # meters
    VACANCY_SCORE_SIGMA = 0.07 # tuned
    DISTANCE_SCORE_SIGMA = 0.2 # tuned

    def __init__(self):
        super().__init__('tiago_reasoner_node')
        
        # Initialize subscriber classes
        self.gdrnet_sub = GDRNetInput(self)
        self.fused_sub = FusedSentenceInput(self)
        
        # Initialize publisher of reasoning result to motion_stack
        self.pub = self.create_publisher(HRICommand, 'tiago_hri_output', 10)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5558")

        # Create feedback publisher for signaling the start of new episode in 
        # the modality fusion and gesture sequence detection
        self.start_episode_pub = self.create_publisher(
            Bool,
            'reasoner/start_episode',
            10
        )

        # Set up ZeroMQ subscriber for motion execution feedback on port 5559
        self.feedback_context = zmq.Context()
        self.motion_execution_feedback_sub = self.feedback_context.socket(zmq.SUB)
        self.motion_execution_feedback_sub.connect("tcp://localhost:5559")
        self.motion_execution_feedback_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Create a lock for handling shared variable access
        self.waiting_lock = threading.Lock()

        # Bool signaling waiting for the end of motion execution
        self.waiting_for_motion_execution = False
        
        # Launch the feedback listener thread
        self.feedback_thread = threading.Thread(target=self.motion_feedback_listener)
        self.feedback_thread.daemon = True
        self.feedback_thread.start()
        
        # Create timer for main processing loop
        self.timer = self.create_timer(0.2, self.main_loop) # speedup?
        
        self.get_logger().info('Reasoner node initialized')

    def is_waiting_for_motion_execution(self):
        """
        Returns True if waiting for motion execution
        """
        with self.waiting_lock:
            return self.waiting_for_motion_execution

    def signal_new_episode(self):
        """
        Publish a single Bool(True) on 'reasoner/start_episode'
        to tell upstream nodes they may start a fresh episode
        """
        msg = Bool(data=True)
        self.start_episode_pub.publish(msg)
        self.get_logger().info("new episode signal sent")

    def motion_feedback_listener(self):
        """
        This thread continuously listens for feedback messages from the motion execution node.
        When a valid message (e.g., with status "success", "failed", or "canceled") is received,
        it updates the shared variable using a lock.
        """
        while rclpy.ok():
            try:
                # Try receiving message without blocking
                message = self.motion_execution_feedback_sub.recv_string(flags=zmq.NOBLOCK)
                self.get_logger().info("Received motion feedback: " + message)
                data = json.loads(message)
                if data.get("status") in ["success", "failed", "canceled"]:
                    with self.waiting_lock:
                        self.waiting_for_motion_execution = False
                    self.get_logger().info("Motion execution ended with status: " + data.get("status"))

            except zmq.Again:
                # No message available, sleep briefly then continue polling
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error("Feedback listener error: " + str(e))
        
    def main_loop(self):
        """
        Main loop of reasoning logic
        Processes the episode command by command, after each command processed
        publish it to motion_stack and wait for movement result
        """
        # Wait until the motion is finished
        with self.waiting_lock:
            if self.waiting_for_motion_execution:
                self.get_logger().info("Waiting for motion execution to finish ...")
                return

        # Wait for the fusion data to arrive, GDRN arrives periodically
        if not self.fused_sub.has_episode():
            self.get_logger().warn("Waiting for fusion data ...")
            self.fused_sub._enabled = True
            self.gdrnet_sub.enable  = True
            return
        
        # Load the received data
        episode = self.fused_sub.get_episode()
        gdrn_objs = self.gdrnet_sub.get_objects()

        self.get_logger().info(f"Fusion data: {episode}")
        self.get_logger().info(f"GDRN data:   {gdrn_objs}")

        self.get_logger().info(f"received steps :{len(episode)}")

        # Process the episode command by command
        for idx, step in enumerate(episode):
            # determine the correct handler based on the action type
            verb = step["action"].lower()
            handler_name = self.ACTION_HANDLERS.get(verb)

            if handler_name is None:
                self.get_logger().error(f"No handler for action “{verb}”")
                return

            handler = getattr(self, handler_name)

            # Process the data using the handler
            self.get_logger().info(f"Processing step {idx+1}: {verb}")
            probs_dict = handler(step, gdrn_objs)

            if not probs_dict: # reasoning failed somewhere
                self.get_logger().warn(f"Reasoning stopped due to logic.")
                return
            
            # Publish the results of reasoning to motion_stack
            self.get_logger().info(f"Publishing commands for step {idx+1}.")

            self.publish_results(
                action=step["action"],
                probs_dict=probs_dict,
                gdrn_objs=gdrn_objs
            )

            # wait for motion execution to finish
            while self.is_waiting_for_motion_execution():
                self.get_logger().info(f"Waiting for motion execution...")
                time.sleep(0.5) # speedup?

        # Signal a start of a new episode, processing complete
        self.get_logger().info(f"Episode processing complete.")
        self.signal_new_episode()
        time.sleep(0.2) # speedup?

    def _handle_pick(self, step, gdrn_objs):
        """
        Handler for 'pick' command action type
        Decides the target object based on the fused and gdrn data
        
        Input:
            step: fused elementary command dict
            gdrn_objs: list with dict of gdrn objects

        Returns:
            probs_dict: dict with keys specifying the target objects and their probabilities
        """
        # Determine the processing type based on the action_param and number of gestures
        action_param = step["action_param"]
        gestures = step["gestures"]

        if action_param != "null":
            # action param present, it specifies the reference object, the gesture is about it
            if action_param not in self.PICK_ALLOWED_RELATIONS:
                self.get_logger().warn(f"Unknown/unsuported action param: {action_param}")
                return
            
            self.get_logger().info(f"Action param: {action_param}")
            
            if len(gestures) == 0:
                # reference object should be specified by object_param in language
                self.get_logger().info(f"Reference object specified in language only")
                probs_dict = self.evaluate_reference_language(step, gdrn_objs)

            elif len(gestures) == 1:
                # reference object is specified by pointing gesture
                self.get_logger().info(f"Reference object specified by language and gesture")
                probs_dict = self.evaluate_reference_language_gesture(step, gdrn_objs)

            else:
                self.get_logger().warn(f"Undefined number of gestures ({len(gestures)}), aborting...")
                return
        
        else:
            # no action param -> specification is about the target object
            if len(gestures) == 0:
                # target object should be specified in language
                self.get_logger().info(f"Target object specified in language only")
                probs_dict = self.evaluate_target_language(step, gdrn_objs)

            elif len(gestures) == 1:
                # target object is specified by pointing gesture
                self.get_logger().info(f"Target object specified by language and gesture")
                probs_dict = self.evaluate_target_language_gesture(step, gdrn_objs)

            else:
                self.get_logger().warn(f"Undefined number of gestures ({len(gestures)}), aborting...")
                return
            
        # Check the reasoning result
        if probs_dict is None:
            self.get_logger().warn("[pick] reasoning failed...")
            return None
        
        self.get_logger().info(f"Final probs: {probs_dict}")
        return probs_dict

    def evaluate_target_language(self, step, gdrn_objs):
        """
        'pick' reasoning when ONLY language is present
        and action_param == 'null'.

        Returns
            probs_dict : {"names": [...], "obj_probs": np.ndarray}  |  None on ambiguity
        """

        # Prepare gdrn data to more favourable structure
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        self.get_logger().info(f"gdrn_dict: {gdrn_dict}")

        obj_names = gdrn_dict["names"]
        obj_confs  = np.array(gdrn_dict["confidences"])

        # Create language mask for target object
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"lang_dict: {lang_dict}")

        target_mask = self.get_language_mask(gdrn_dict, lang_dict, object_type="target")
        self.get_logger().info(f"target_mask: {target_mask}")
        
        # Language filtering test
        n_hits = int(target_mask.sum())
        if n_hits == 0:                              # nothing matched
            self.get_logger().warn(
                "Language matches no target object."
            )
            return None
        
        # P_obj = obj_confs . target_mask
        probs = np.zeros(len(obj_names))
        probs[target_mask] = obj_confs[target_mask] # the object with matching params and highest gdrn conf is winner
        probs = probs / probs.sum()

        # Create a resulting dict
        probs_dict = {
            "names": obj_names,
            "obj_probs": probs
        }

        return probs_dict

    def evaluate_target_language_gesture(self, step, gdrn_objs):
        """
        'pick' reasoning when BOTH gesture and language are present
        and action_param == 'null'.

        Returns
            probs_dict : {"names": [...], "obj_probs": np.ndarray}  |  None on ambiguity
        """

        # Load GDRN and deictic data 
        gesture   = step["gestures"][0]               # exactly one gesture here
        gest_dict = self.get_gesture_objs_dict(gesture)
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)

        self.get_logger().info(f"gesture_dict: {gest_dict}")
        self.get_logger().info(f"gdrn_dict   : {gdrn_dict}")

        # Align gesture probabilities to GDRN order 
        aligned = self.align_gestures_to_gdrn(gest_dict, gdrn_dict)
        if aligned is None:
            return None

        gdrn_names  = aligned["gdrn_names"]
        gdrn_confs  = aligned["gdrn_confs"]
        gest_probs  = aligned["gest_probs"]

        # Create target language mask
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"lang_dict: {lang_dict}")

        target_mask = self.get_language_mask(gdrn_dict, lang_dict, object_type="target")
        self.get_logger().info(f"target_mask: {target_mask}")

        if not target_mask.any():
            # Language didn’t match anything, command ambiguous
            self.get_logger().warn("Language matches no target object.")
            return None

        # P_obj = gesture . gdrn . language -------------------------
        probs = np.zeros(len(gdrn_names))
        probs[target_mask] = gest_probs[target_mask] * gdrn_confs[target_mask]

        if probs.sum() == 0:
            self.get_logger().warn("Probabilities sum to zero after masking.")
            return None

        # Create resulting probs dict
        probs /= probs.sum()
        probs_dict = {
            "names": gdrn_names,
            "obj_probs": probs
        }
        
        return probs_dict
    
    def evaluate_reference_language(self, step, gdrn_objs):
        """
        'pick' reasoning when ONLY language is present *and*
        action_param is a spatial relation

        Returns
            probs_dict : {"names": [...], "obj_probs": np.ndarray} | None on ambiguity
        """
        # Check relation plausability
        relation = step["action_param"].lower()
        if relation not in self.PICK_ALLOWED_RELATIONS:
            self.get_logger().warn(f"Relation '{relation}' not supported.")
            return None

        # Load GDRN data
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        self.get_logger().info(f"gdrn_dict: {gdrn_dict}")

        gdrn_names = gdrn_dict["names"]
        gdrn_confs = np.array(gdrn_dict["confidences"])  

        # more than 1 object in scene needed for relation to make sence
        N = len(gdrn_names)
        if N < 2:
            self.get_logger().warn("Less than two objects - relation meaningless.")
            return None

        # Language proccesing 
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"lang_dict: {lang_dict}")

        # 1. Create masks for both reference and target object
        ref_mask = self.get_language_mask(gdrn_dict, lang_dict, object_type="reference")
        if not ref_mask.any():
            self.get_logger().warn("Language matches no reference object.")
            return None
        self.get_logger().info(f"ref_mask: {ref_mask}")
        
        target_mask = self.get_language_mask(gdrn_dict, lang_dict, object_type="target")
        if not target_mask.any():
            self.get_logger().warn("Language matches no target object.")
            return None
        self.get_logger().info(f"target_mask: {target_mask}")

        # 2. P(reference) proportional to GDRN confidence x language reference params
        P_ref = np.zeros(N)
        P_ref[ref_mask] = gdrn_confs[ref_mask]
        P_ref /= P_ref.sum()

        self.get_logger().info(f"Language reference probabilities: {gdrn_names} - {P_ref}")

        # 3. P(target) = sum(P(target | ref_i) for each i)
        scores = self.collect_target_probs(
            P_ref,
            relation,
            target_mask,
            gdrn_dict,
            gdrn_confs
        )

        
        if scores is None:
            return None

        # Create resultion probs dict
        probs_dict = {"names": gdrn_names, "obj_probs": scores}
        return probs_dict

    def evaluate_reference_language_gesture(self, step, gdrn_objs):
        """
        'pick' reasoning when we have a relation in language
        and exactly one pointing gesture (reference disambiguation)

        Returns
            probs_dict : {"names": [...], "obj_probs": np.ndarray} | None
        """

        # Check relation plausability
        relation = step["action_param"].lower()
        if relation not in self.PICK_ALLOWED_RELATIONS:
            self.get_logger().warn(f"Relation '{relation}' not supported.")
            return None

        # Load GDRN and gesture data
        gesture = step["gestures"][0]
        gest_dict = self.get_gesture_objs_dict(gesture)
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)

        self.get_logger().info(f"gdrn_dict: {gdrn_dict}")
        self.get_logger().info(f"gest_dict: {gest_dict}")

        # relation requires 2 objects present
        N = len(gdrn_dict["names"])
        if N < 2:
            self.get_logger().warn("Less than two objects - relation meaningless.")
            return None

        # Align gesture probabilities to GDRN order
        aligned = self.align_gestures_to_gdrn(gest_dict, gdrn_dict)
        if aligned is None:
            return None

        gdrn_names  = aligned["gdrn_names"]
        gdrn_confs  = aligned["gdrn_confs"]
        gest_probs  = aligned["gest_probs"]

        # Create language masks for target and reference
        lang_dict   = self.get_lang_objs_dict(step)
        self.get_logger().info(f"lang_dict: {lang_dict}")

        ref_mask = self.get_language_mask(gdrn_dict, lang_dict, object_type="reference")
        if not ref_mask.any():
            self.get_logger().warn("Language matches no reference object.")
            return None
        self.get_logger().info(f"ref_mask: {ref_mask}")
        
        target_mask = self.get_language_mask(gdrn_dict, lang_dict, object_type="target")
        if not target_mask.any():
            self.get_logger().warn("Language matches no target object.")
            return None
        self.get_logger().info(f"target_mask: {target_mask}")

        # P_ref(i) = gesture . conf . language
        alpha = 2.0 # coeficient sharpening the probability of the reference from pointing
        P_ref = (gest_probs ** alpha) * gdrn_confs * ref_mask

        if P_ref.sum() == 0:
            self.get_logger().warn("Gesture points at nothing that matches language.")
            return None
        P_ref /= P_ref.sum()

        # Loop over reference candidates
        scores = self.collect_target_probs(
            P_ref,
            relation,
            target_mask,
            gdrn_dict,
            gdrn_confs
        )

        if scores is None:
            return None

        # Create resulting probs dict
        probs_dict = {"names": gdrn_names, "obj_probs": scores}
        return probs_dict

    def _handle_place(self, step, gdrn_objs):
        """
        Handler for 'place' command action type
        Decides the place pose based on the fused and gdrn data
        This action type requires to have an action parameter
        
        Input:
            step: fused elementary command dict
            gdrn_objs: list with dict of gdrn objects

        Returns:
            probs_dict: dict with keys specifying the place poses and their probabilities
        """
        action_param = step["action_param"]
        if action_param == "null":
            self.get_logger().warn(f"Unknown action param: {action_param}")
            return
        
        # Filter allowed params
        if action_param not in self.PLACE_ALLOWED_RELATIONS:
            self.get_logger().warn(f"Unknown action param: {action_param}")
            return
        
        self.get_logger().info(f"Action param: {action_param}")
        gestures = step["gestures"]

        # place can only have 0 or 1 gesture
        if len(gestures) > 1: 
            self.get_logger().warn(f"Invalid number of gestures for action 'place': {len(gestures)}")
            return

        if action_param == "here" and len(gestures) != 1:
            self.get_logger().warn(f"Invalid number of gestures for action param '{action_param}': {len(gestures)}")
            return

        if action_param == "here":
            # Place pose specified directly from the gesture
            self.get_logger().info(f"Determining target place pose from gesture...")
            place_pose_probs = self.determine_place_pose_here(step, gdrn_objs)
            
        elif len(gestures) == 0:
            # reference object should be specified by object_param in language, from action param then determine target place pose
            self.get_logger().info(f"Reference object specified in language only")
            place_pose_probs = self.determine_place_pose_language(step, gdrn_objs)

        elif len(gestures) == 1:
            # reference object is specified by language and gesture, from action param then specify the target pose
            self.get_logger().info(f"Reference object specified by language and gesture")
            place_pose_probs = self.determine_place_pose_language_gesture(step, gdrn_objs)

        else:
            self.get_logger().warn(f"Something wrong...")
            return
        
        # Check the reasoning result
        if place_pose_probs is None:
            self.get_logger().warn("[place] reasoning failed...")
            return None

        self.get_logger().info(f"Final probs: {place_pose_probs}")
        return place_pose_probs
    
    def determine_place_pose_here(
        self,
        step,
        gdrn_objs,
        fallback_table_z: float = FALLBACK_TABLE_HEIGHT
    ):
        """
        'place' reasoning when we have a relation 'here'
        and exactly one pointing gesture (reference disambiguation)        
        Computes the 3-D pose that corresponds to the pointing gesture
        The pointing gesture gives two 3-D points `p0`, `p1`
        (line through the wrist and fingertip).  
        We intersect that line with the horizontal plane *z = z_table*.
        If the line is (almost) parallel to the plane - an intersection does
        not exist; we fall back to the fingertip projection.

        Returns
            probs_dict {"places": [place_pose : [x, y, z]], "place_probs": [1.0]} 
                Cartesian coordinates in robot base, place pose certain from gesture
                Orientation is not decided here - we return only XYZ.

        """
        # Determine table height
        if gdrn_objs:
            z_vals = [obj["position"][2] for obj in gdrn_objs]
            z_table = float(np.mean(z_vals))
        else:
            z_table = fallback_table_z

        # Clamp the z_table at a minimum value for the gripper not destroying the table
        if z_table < fallback_table_z:
            z_table = fallback_table_z
        self.get_logger().info(f"[place-here]  table-z ≈ {z_table:.3f} m")

        # Construct a line
        g = step["gestures"][0]
        p0 = np.array([
            g["line_points"][0].x,
            g["line_points"][0].y,
            g["line_points"][0].z
        ], dtype=float)
        p1 = np.array([
            g["line_points"][1].x,
            g["line_points"][1].y,
            g["line_points"][1].z
        ], dtype=float)

        # direction vector
        dir_v  = p1 - p0 

        # If direction has almost zero z-component, the line is parallel
        eps = 1e-5
        if abs(dir_v[2]) < eps:
            self.get_logger().warn("[place-here] pointing parallel to table - "
                                "using fingertip projection.")
            t = 0.0 # take p0
        else:
            t = (z_table - p0[2]) / dir_v[2]

        # Compute the intersection (projection)
        place_p = p0 + t * dir_v

        # The resulting intercesction is the place pose
        place_pose = place_p.tolist()
        self.get_logger().info(
            f"[place-here]  chosen pose (xyz) = {np.round(place_pose,3)}"
        )

        # wrap for publish_results
        return {
            "places": [place_pose],
            "place_probs" : np.array([1.0])
        }
    
    def determine_place_pose_language(
        self,
        step: dict,
        gdrn_objs: list[dict],
        offset: float = REFERENCE_POSE_OFFSET,
        fallback_table_z: float = FALLBACK_TABLE_HEIGHT
    ):
        """
        'place' reasoning when we have a spatial relation
        and no pointing gesture

        Returns
            probs_dict { "places": [[x,y,z] …], "place_probs": np.ndarray }
            or None  if the language is ambiguous / no valid place exists.
        """
        # Filter allowed relations
        relation = step["action_param"].lower()
        if relation not in self.PLACE_SUPPORTED_SPATIAL_RELATIONS:
            self.get_logger().warn(f"[place-lang] relation “{relation}” unsupported")
            return None

        # Load GDRN data
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        if len(gdrn_dict["names"]) == 0:
            self.get_logger().warn("[place-lang] no objects on table - cannot use relation")
            return None
        
        self.get_logger().info(f"gdrn_dict: {gdrn_dict}")

        # Create reference language mask
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"lang_dict: {lang_dict}")

        ref_mask = self.get_language_mask(gdrn_dict, lang_dict, "reference")
        if not ref_mask.any():
            self.get_logger().warn("[place-lang] language matches no reference object")
            return None

        # P_ref = detection_conf · language_mask
        P_ref = np.asarray(gdrn_dict["confidences"]) * ref_mask
        if P_ref.sum() == 0:
            self.get_logger().warn("[place-lang+gest] language does not describe valid reference")
            return None
        P_ref /= P_ref.sum()

        # Compute table height
        zs = [p[2] for p in gdrn_dict["positions"]]
        table_z = max(float(np.mean(zs)), fallback_table_z)

        # Determine the probs for poses
        return self.collect_place_poses_from_ref(
            relation,
            P_ref,
            gdrn_dict,
            offset,
            table_z
        )
    
    def determine_place_pose_language_gesture(
        self,
        step: dict,
        gdrn_objs: list[dict],
        offset: float = REFERENCE_POSE_OFFSET,
        fallback_table_z: float = FALLBACK_TABLE_HEIGHT,
    ):
        """
        'place' reasoning when we have a spatial relation
        and exactly one pointing gesture (reference disambiguation)

        Returns
            probs_dict  { "places": [[x,y,z] …], "probs": np.ndarray }
            or None  if the reasoning fails.
        """
        # Check allowed relation and gesture presence
        relation = step["action_param"].lower()
        if relation not in self.PLACE_SUPPORTED_SPATIAL_RELATIONS:
            self.get_logger().warn(f"[place-lang+gest] relation “{relation}” unsupported")
            return None

        if not step["gestures"]:
            self.get_logger().warn("[place-lang+gest] no gesture found")
            return None
        gesture = step["gestures"][0]

        # Load GDRN and gesture data
        gest_dict = self.get_gesture_objs_dict(gesture)
        self.get_logger().info(f"gest_dict: {gest_dict}")
        
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        if len(gdrn_dict["names"]) == 0:
            self.get_logger().warn("[place-lang] no objects on table - cannot use relation")
            return None
        self.get_logger().info(f"gdrn_dict: {gdrn_dict}")

        # Align gesture probabilities to GDRN order
        aligned = self.align_gestures_to_gdrn(gest_dict, gdrn_dict)
        if aligned is None:
            return None
        
        gdrn_confs = aligned["gdrn_confs"]
        gest_probs = aligned["gest_probs"]

        # Create language reference mask
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"lang_dict: {lang_dict}")

        ref_mask = self.get_language_mask(gdrn_dict, lang_dict, "reference")
        if not ref_mask.any():
            self.get_logger().warn("[place-lang+gest] language matches no reference")
            return None

        # P_ref = gesture . conf . language_mask
        P_ref = gest_probs * gdrn_confs * ref_mask
        if P_ref.sum() == 0:
            self.get_logger().warn("[place-lang+gest] gesture conflicts with language")
            return None
        P_ref /= P_ref.sum()

        # compute table height
        zs = [p[2] for p in gdrn_dict["positions"]]
        table_z = max(float(np.mean(zs)), fallback_table_z)

        # Determine the probs for poses
        return self.collect_place_poses_from_ref(
            relation,
            P_ref,
            gdrn_dict,
            offset,
            table_z
        )

    def _handle_move(self, step, gdrn_objs):
        """
        Handler for 'move' command action type
        Decides the target object and place pose based on the fused and gdrn data
        This action type requires to have an action parameter
        
        Input:
            step: fused elementary command dict
            gdrn_objs: list with dict of gdrn objects

        Returns:
            probs_dict: dict with keys specifying the target objects, place poses and their probabilities separately
        """
        action_param = step["action_param"]
        if action_param == "null":
            self.get_logger().warn(f"[move] Action '{action_param}' requires action parameter.")
            return None
        
        gestures = step["gestures"]

        if len(gestures) == 0 and action_param != "here":
            # the target object and place pose described both in language only
            self.get_logger().info(f"Target object and place pose specified in language only")
            probs_dict = self.process_move_language(step, gdrn_objs)
        
        elif len(gestures) == 1:
            # the target object or place pose described in language and gesture, the other only language
            self.get_logger().info(f"Target object or place pose specified in language only")
            probs_dict = self.process_move_language_single_gesture(step, gdrn_objs)

        elif len(gestures) == 2:
            # both target object and place pose described in language and gesture
            self.get_logger().info(f"Target object and place pose specified in language and gesture")
            probs_dict = self.process_move_language_gestures(step, gdrn_objs)

        else:
            self.get_logger().info(f"Received {len(gestures)} gestures with action param {action_param}, aborting.")
            return None
        
        # Check reasoning result
        if probs_dict is None:
            self.get_logger().warn("[place] reasoning failed...")
            return None

        self.get_logger().info(f"Final probs: {probs_dict}")
        return probs_dict
    
    def process_move_language(
        self,
        step: dict,
        gdrn_objs: list[dict],
        offset: float = REFERENCE_POSE_OFFSET,
        fallback_table_z: float = FALLBACK_TABLE_HEIGHT,
    ):
        """
        'move' reasoning when we have a spatial relation
        and no pointing gesture

        1.  Choose the target OBJECT target based on language only
            P(obj) = language_mask(target) · gdrn_conf

        2.  Choose the place pose for that object
            - choose reference candidates with language_mask(reference)
            - for each ref build candidate pose complying with spatial relation
            - score pose = P(ref) · vacancy_score(pose)

        Returns
            {
                "names"       : [list of gdrn object names]
                "obj_probs"   : np.ndarray
                "places"      : [[x,y,z], …]
                "place_probs" : np.ndarray
            } | None on ambiguity / failure
        """
        # Check action param validity
        relation = step["action_param"].lower()
        if relation not in self.PLACE_SUPPORTED_SPATIAL_RELATIONS:
            self.get_logger().warn(f"[move-lang] unsupported relation “{relation}”")
            return None

        # Load GDRN data
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        if len(gdrn_dict["names"]) == 0:
            self.get_logger().warn("[move-lang] no objects visible")
            return None
        self.get_logger().info(f"[move-lang] gdrn_dict: {gdrn_dict}")

        # 1. Create target object probs from language mask and GDRN confidences
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"[move-lang] lang_dict: {lang_dict}")

        target_mask = self.get_language_mask(gdrn_dict, lang_dict, "target")
        if not target_mask.any():
            self.get_logger().warn("[move-lang] language matches no target object")
            return None

        gdrn_confs = np.asarray(gdrn_dict["confidences"])
        P_obj = gdrn_confs * target_mask
        P_obj /= P_obj.sum()

        # 2. Compute reference prior P_ref(i)
        ref_mask = self.get_language_mask(gdrn_dict, lang_dict, "reference")
        if not ref_mask.any():
            self.get_logger().warn("[move-lang] language matches no reference object")
            return None

        P_ref = gdrn_confs * ref_mask
        P_ref /= P_ref.sum()

        # Compute table height
        zs = [p[2] for p in gdrn_dict["positions"]]
        table_z = max(float(np.mean(zs)), fallback_table_z)

        # Determine the probs for poses
        place_dict = self.collect_place_poses_from_ref(
            relation,
            P_ref,
            gdrn_dict,
            offset,
            table_z
        )

        if place_dict is None:
            return None

        # assemble the probs_dict
        return {
            "names"       : gdrn_dict["names"],
            "obj_probs"   : P_obj,
            "places"      : place_dict["places"],
            "place_probs" : place_dict["place_probs"],
        }

    def process_move_language_single_gesture(
        self,
        step: dict,
        gdrn_objs: list[dict],
        offset: float = REFERENCE_POSE_OFFSET,
        fallback_table_z: float = FALLBACK_TABLE_HEIGHT,
    ):
        """
        'move' reasoning when we have only one pointing gesture
        based on the gesture role decide the use of gesture probability
        for target object or place pose

        1.  Choose the target OBJECT target 
        2.  Choose the place pose for that object

        Returns
            {
                "names"       : [list of gdrn object names]
                "obj_probs"   : np.ndarray
                "places"      : [[x,y,z], …]
                "place_probs" : np.ndarray
            } | None on ambiguity / failure

        Supported cases
            1) gesture_role == "object"
                Move *this* banana left of the can.
            2) gesture_role == "place"
                Move the banana left of *this* can.
                Move the banana *here*.
        """
        # Check determined gesture role
        if "gesture_role" not in step or step["gesture_role"] not in self.MOVE_SUPPORTED_GESTURE_ROLES:
            self.get_logger().warn("[move-lang+1g] missing/unknown gesture_role tag")
            return None
        
        role = step["gesture_role"]
        gesture = step["gestures"][0]
        relation = step["action_param"].lower()

        # Load gdrn and language data
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        if len(gdrn_dict["names"]) == 0:
            self.get_logger().warn("[move-1g] no GDRN objects visible")
            return None
        self.get_logger().info(f"[move-1g] gdrn_dict: {gdrn_dict}")

        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"[move-1g] lang_dict: {lang_dict}")

        gdrn_confs = np.asarray(gdrn_dict["confidences"])

        # 1. Determine the target object based on the gesture role
        if role == "object":
            # Gesture helps with the target object
            self.get_logger().info("[move-1g] Gesture specifies the target object")

            target_mask = self.get_language_mask(gdrn_dict, lang_dict, "target")
            if not target_mask.any():
                self.get_logger().warn("[move-1g] language does not describe target")
                return None

            aligned = self.align_gestures_to_gdrn(
                self.get_gesture_objs_dict(gesture), gdrn_dict
            )
            if aligned is None:
                return None
            gest_probs = aligned["gest_probs"]

            P_obj = gest_probs * gdrn_confs * target_mask
            if P_obj.sum() == 0:
                self.get_logger().warn("[move-1g] gesture incompatible with language target")
                return None
            P_obj /= P_obj.sum()

        else:
            # Compute the target probs without gesture
            target_mask = self.get_language_mask(gdrn_dict, lang_dict, "target")
            if not target_mask.any():
                self.get_logger().warn("[move-1g] language does not describe target")
                return None
            P_obj = gdrn_confs * target_mask
            P_obj /= P_obj.sum()

        # 2. Determine the place pose based on the action param and gesture role
        if relation == "here":
            # Explicit pointing defines the pose, needed 'place' gesture role
            if role != "place":
                self.get_logger().warn("[move-1g] need the *place* gesture for “here”")
                return None
            self.get_logger().info("[move-1g] Gesture specifies the place pose")

            # Determine the place pose directly
            pose_dict = self.determine_place_pose_here(
                {"gestures":[gesture]}, gdrn_objs, fallback_table_z
            )
            if pose_dict is None:
                return None
            
            place_poses = pose_dict["places"]
            place_probs = pose_dict["place_probs"]

        else:
            # Place pose determined from relation
            if relation not in self.PLACE_SUPPORTED_SPATIAL_RELATIONS:
                self.get_logger().warn(f"[move-1g] unsupported relation “{relation}”")
                return None

            if role == "place":
                # Gesture points at reference object
                self.get_logger().info("[move-1g] Gesture specifies the reference object")
                
                ref_mask = self.get_language_mask(gdrn_dict, lang_dict, "reference")
                if not ref_mask.any():
                    self.get_logger().warn("[move-1g] language lacks reference object")
                    return None

                aligned = self.align_gestures_to_gdrn(
                    self.get_gesture_objs_dict(gesture), gdrn_dict
                )
                if aligned is None:
                    return None
                gest_probs = aligned["gest_probs"]

                P_ref = gest_probs * gdrn_confs * ref_mask
                if P_ref.sum() == 0:
                    self.get_logger().warn("[move-1g] gesture incompatible with reference")
                    return None
                P_ref /= P_ref.sum()

            else:
                # Determine place pose from language only
                ref_mask = self.get_language_mask(gdrn_dict, lang_dict, "reference")
                if not ref_mask.any():
                    self.get_logger().warn("[move-1g] language lacks reference object")
                    return None
                P_ref = gdrn_confs * ref_mask
                P_ref /= P_ref.sum()

            # Compute table height
            zs       = [p[2] for p in gdrn_dict["positions"]]
            table_z  = max(float(np.mean(zs)), fallback_table_z)

            # Determine place pose probs
            place_dict = self.collect_place_poses_from_ref(
                relation, P_ref, gdrn_dict, offset, table_z
            )
            if place_dict is None:
                return None
            place_poses  = place_dict["places"]
            place_probs  = place_dict["place_probs"]

        # Create a combined probs dict
        return {
            "names"       : gdrn_dict["names"],
            "obj_probs"   : P_obj,
            "places"      : place_poses,
            "place_probs" : place_probs,
        }

    def process_move_language_gestures(
        self,
        step: dict,
        gdrn_objs: list[dict],
        offset: float = REFERENCE_POSE_OFFSET,
        fallback_table_z: float = FALLBACK_TABLE_HEIGHT,
    ):
        """
        'move' reasoning when we have two pointing gestures
        Both the target and reference objects are also determined by gesture

        1.  Choose the target OBJECT target 
        2.  Choose the place pose for that object
        """
        # Check gestures number
        gestures = step["gestures"]
        if len(gestures) != 2:
            self.get_logger().warn("[move-lang+2g] exactly 2 gestures required")
            return None
        gesture_target, gesture_ref = gestures

        relation = step["action_param"].lower()

        # Load GDRN data
        gdrn_dict = self.get_gdrn_objs_dict(gdrn_objs)
        if len(gdrn_dict["names"]) == 0:
            self.get_logger().warn("[move-lang+2g] no GDRN objects in view")
            return None
        self.get_logger().info(f"[move-lang+2g] gdrn_dict: {gdrn_dict}")

        # Create target language mask
        lang_dict = self.get_lang_objs_dict(step)
        self.get_logger().info(f"[move-lang+2g] lang_dict: {lang_dict}")

        target_mask = self.get_language_mask(gdrn_dict, lang_dict, "target")
        if not target_mask.any():
            self.get_logger().warn("[move-lang+2g] language does not specify target")
            return None

        # Align target gesture to GDRN order
        aligned_gesture_target = self.align_gestures_to_gdrn(
            self.get_gesture_objs_dict(gesture_target), gdrn_dict
        )
        if aligned_gesture_target is None:
            return None
        gesture_target_probs = aligned_gesture_target["gest_probs"]
        gdrn_confs = aligned_gesture_target["gdrn_confs"]

        # P_obj = gesture . confidence . lang_mask
        P_obj = gesture_target_probs * gdrn_confs * target_mask
        if P_obj.sum() == 0:
            self.get_logger().warn("[move-lang+2g] target gesture incompatible")
            return None
        P_obj /= P_obj.sum()

        # Determine the place pose
        if relation == "here":
            # Place pose directly specified by gesture
            self.get_logger().info(f"[move-lang+2g] Determining place pose from gesture")
            
            pose_dict = self.determine_place_pose_here(
                {"gestures":[gesture_ref]}, gdrn_objs, fallback_table_z
            )
            if pose_dict is None:
                return None
            place_poses  = pose_dict["places"]
            place_probs  = pose_dict["place_probs"]

        elif relation in self.PLACE_SUPPORTED_SPATIAL_RELATIONS:
            # Place pose specified by relation to reference object
            self.get_logger().info(f"[move-lang+2g] Determining place pose reference target pose")
            
            # Create reference language mask
            ref_mask = self.get_language_mask(gdrn_dict, lang_dict, "reference")
            if not ref_mask.any():
                self.get_logger().warn("[move-lang+2g] language does not specify reference")
                return None

            # Align reference gesture to GDRN order
            aligned_gesture_ref = self.align_gestures_to_gdrn(
                self.get_gesture_objs_dict(gesture_ref), gdrn_dict
            )
            if aligned_gesture_ref is None:
                return None
            gesture_ref_probs = aligned_gesture_ref["gest_probs"]

            # P_ref = gesture . confidence . lang_mask
            P_ref = gesture_ref_probs * gdrn_confs * ref_mask
            if P_ref.sum() == 0:
                self.get_logger().warn("[move-lang+2g] reference gesture incompatible")
                return None
            P_ref /= P_ref.sum()

            # Determine table height
            zs = [p[2] for p in gdrn_dict["positions"]]
            table_z = max(float(np.mean(zs)), fallback_table_z)

            # Determine place pose probs
            place_dict = self.collect_place_poses_from_ref(
                relation, P_ref, gdrn_dict, offset, table_z
            )
            if place_dict is None:
                return None
            place_poses  = place_dict["places"]
            place_probs  = place_dict["place_probs"]

        else:
            # Unsuported relation, do nothing
            self.get_logger().warn(f"[move-lang+2g] unsupported relation “{relation}”")
            return None

        # Create combined probs_dict
        return {
            "names"       : gdrn_dict["names"],
            "obj_probs"   : P_obj,
            "places"      : place_poses,
            "place_probs" : place_probs,
        }


    # --- HELPER FUNCTIONS PART ---
    def strip_name(self, full_name: str) -> str:
      """
      Convert GDRN object IDs to plain object names

      Examples
        strip_name("011_banana_1") -> 'banana'
        strip_name("003_tomato_soup_can_01") -> 'tomato_soup_can'
      """
      # Remove the leading numeric code
      name = re.sub(r'^\d+_', '', full_name)
      # Remove the trailing instance index
      name = re.sub(r'_\d+$', '', name)

      return name

    def get_lang_objs_dict(self, step):
        """
        Returns the dict containing only the object names and params from language
        """
        return {
            "names":  step["objects"],
            "params": step["objects_param"],
        }

    def get_gdrn_objs_dict(self, gdrn_objs):
        """
        Returns one dict with info about all GDRN objects
        """
        obj_names = []
        obj_confidences = []
        obj_colors = []
        for obj in gdrn_objs:
            obj_names.append(obj["name"])
            obj_confidences.append(obj["confidence"])
            obj_colors.append(obj["color"])

        gdrn_probs_dict = {
            "names": obj_names,
            "confidences": obj_confidences,
            "colors": obj_colors,
            "positions": [obj["position"] for obj in gdrn_objs]
        }
        return gdrn_probs_dict
    
    def get_gesture_objs_dict(self, gesture: dict):
        """
        Returns dict of objects and their probs of being pointed at 
        from the gesture dict
        """
        gesture_probs = gesture["object_likelihoods"]
        gesture_probs = gesture_probs / np.sum(gesture_probs) # normalize likelihoods from the gesture

        object_names = gesture["object_names"]

        gesture_probs_dict = {
            "names": object_names,
            "probs": gesture_probs
        }
        return gesture_probs_dict
    
    def get_language_mask(self, gdrn_dict, lang_dict, object_type="target"):
        """
        Creates language mask for objects in GDRN, based on object_type

        Input:
            gdrn_dict - dict with info about all GDRN objects
            lang_dict - dict with info about target and reference objects from language
            object_type - string determining the type of mask
                object_type='target' - masking objects based on the target in language
                object_type='reference' - masking objects based on the reference in language

        Returns
            mask - list[bool] marking which objects in GDRN order match the description of object_type object

        """
        gdrn_names = gdrn_dict["names"]
        gdrn_colors = gdrn_dict["colors"]
        N = len(gdrn_names)

        # Specify type of object
        index = None
        if object_type == "target":
            index = 0 # Index of the target object
        elif object_type =="reference":
            index = 1 # Index of the reference object
        else:
            self.get_logger().warn(f"Unknown object type given - '{object_type}'")
            return 

        obj_name  = lang_dict["names"][index]
        obj_param = lang_dict["params"][index]

        # Create language mask
        mask = np.ones(N, dtype=bool)

        # Matching object name
        if obj_name != "null":
            stripped = [self.strip_name(n) for n in gdrn_names]
            mask &= np.array([obj_name in n for n in stripped]) # in for multiword object names when not said entirely
        else:
            self.get_logger().info("Object name is null (accept any).")

        # Matching object color
        if obj_param != "null":
            mask &= np.array([
                (obj_param in c) if isinstance(c, list) else (c == obj_param)
                for c in gdrn_colors
            ])
        else:
            self.get_logger().info("Object param is null (accept any).")

        return mask
    
    def get_relation_mask(self, i_ref, relation, gdrn_dict, tol=1e-2):
        """
        Build a list[bool] mask over all GDRN objects that are in the given
        relation to the reference object at index 'i_ref'

        Input
            i_ref    : int    index of reference object in gdrn_dict
            relation : str    'left' | 'right' | 'front' | 'behind' | 'shape' | 'color'
            gdrn_dict: dict   output of self.get_gdrn_objs_dict()
            tol      : float  tolerance for dot-product side tests

        Returns
            mask : list[bool] marking which objects in GDRN order match the relation with 'ref_i' object
        """
        N     = len(gdrn_dict["names"])
        mask  = np.zeros(N, dtype=bool)

        # If no objects, we dont need to compute anything
        if N == 0:
            return mask

        # Prepare input data
        gdrn_pos = np.array([p[:2] for p in gdrn_dict["positions"]])  # (N,2) x,y
        gdrn_colors = gdrn_dict["colors"]
        gdrn_names_stripped  = [self.strip_name(n) for n in gdrn_dict["names"]]

        ref_pos = gdrn_pos[i_ref]
        relation = relation.lower()

        # Handle the spacial relations 
        # XXX IMPORTANT: the relations are from the ROBOT'S POV.
        if relation in ("left", "right", "front", "behind"):
            dx = gdrn_pos[:, 0] - ref_pos[0]    # +X behind
            dy = gdrn_pos[:, 1] - ref_pos[1]    # +Y left
            if relation == "left":
                mask = dy >  tol
            elif relation == "right":
                mask = dy < -tol
            elif relation == "behind":
                mask = dx >  tol
            elif relation == "front":
                mask = dx < -tol

        # Similarity relations
        elif relation == "shape":
            ref_name = gdrn_names_stripped[i_ref]
            mask = np.array([n == ref_name for n in gdrn_names_stripped])

        elif relation == "color":
            ref_col = gdrn_colors[i_ref]
            def color_match(c):
                """Helper function for comparing colors, some objects have more than one"""
                if isinstance(ref_col, list):
                    return any(col in c for col in ref_col) if isinstance(c, list) else (c in ref_col)
                else:
                    return (ref_col in c) if isinstance(c, list) else (c == ref_col)
            mask = np.array([color_match(c) for c in gdrn_colors])

        # Reference cannot be its own target
        mask[i_ref] = False
        return mask

    def collect_target_probs(
        self,
        P_ref: np.ndarray,              # shape (N,)
        relation: str,
        target_mask: np.ndarray,        # shape (N,)
        gdrn_dict: dict,                # output of get_gdrn_objs_dict
        gdrn_confs: np.ndarray,         # shape (N,)
    ):
        """
        Turn a 'P_ref' probability vector into a probability
        distribution over target objects

        Implements the logic:
            P(target) = Σᵢ  P(ref=i) · P(target | ref=i)

        where  P(target | ref=i)  is computed from
        • relation mask (spacial / similarity)
        • Gaussian distance prior
        • GDRN confidence of targets
        """
        N       = len(P_ref)
        scores  = np.zeros(N)

        gdrn_poses = np.array([p[:2] for p in gdrn_dict["positions"]]) # x, y only

        # Iterate for all possible ref objects
        for i, p_ref in enumerate(P_ref):
            if p_ref == 0:
                # Object i not a plausible reference, skip
                continue                      

            # Combine target mask and relation mask to reference i
            rel_mask   = self.get_relation_mask(i, relation, gdrn_dict)
            candidate  = rel_mask & target_mask
            if not candidate.any():
                continue

            # Compute distance probability coeff.
            dx   = gdrn_poses[:, 0] - gdrn_poses[i, 0]
            dy   = gdrn_poses[:, 1] - gdrn_poses[i, 1]
            distances = np.sqrt(dx**2 + dy**2)

            dist_probs = np.zeros(N)
            dist_probs[candidate] = self.get_distance_probs(distances[candidate])
            if dist_probs.sum() == 0:
                continue
            dist_probs /= dist_probs.sum() # P(· | ref=i, geom)

            # Combine distance coeff with GDRN confidence
            target_prob_from_ref = dist_probs * gdrn_confs
            if target_prob_from_ref.sum() == 0:
                continue
            target_prob_from_ref /= target_prob_from_ref.sum()

            # Accumulate Σᵢ P(ref=i) · P(target=j | ref=i)
            scores += p_ref * target_prob_from_ref

            self.get_logger().info(
                f"Target probabilities from reference {i}: {np.round(target_prob_from_ref,3)}"
            )

        # Check and return scores - P(obj=i)
        if scores.sum() == 0:
            self.get_logger().warn("No valid target found given the relation.")
            return None
        scores /= scores.sum()
        return scores

    def get_vacancy_score(
        self,
        place_xy: np.ndarray,
        obj_xy: np.ndarray,
        sigma: float = VACANCY_SCORE_SIGMA       
    ) -> float:
        """
        Returns a vacancy coeff. in [0,1]; high when the nearest object is far enough.
        Uses score = 1 - exp( -(d_min²)/(2 σ²) )
        """
        if obj_xy.size == 0:
            return 1.0
        dists = np.linalg.norm(obj_xy - place_xy, axis=1)
        d_min = float(dists.min())
        return float(1 - np.exp(-(d_min**2) / (2 * sigma**2)))

    def get_distance_probs(self, distances: list, sigma=DISTANCE_SCORE_SIGMA):
        """
        Returns list of probabilities determined distances using Gaussian distribution and normalized
        Uses score = exp( -(dist²)/(2 σ²) )
        """
        probs = []
        for dist in distances:
            prob = np.exp(-(dist**2) / (2 * sigma**2))
            probs.append(prob)
        probs = probs / np.sum(probs)
        return list(probs)

    def collect_place_poses_from_ref(
        self,
        relation: str,
        P_ref: np.ndarray,
        gdrn_dict: dict,
        offset: float,
        table_z: float,
    ):
        """
        Collects place poses and their probs given the spacial relation

        Input
            relation  : 'left' | 'right' | 'front' | 'behind'
            P_ref     : np.ndarray(N,) prior over reference objects
            gdrn_dict : output of get_gdrn_objs_dict
            offset    : float ofsset of the place pose from the position of reference object
            table_z   : float z-height of the table

        Returns
            probs_dict { "places": [[x,y,z]…], "probs": np.ndarray }  |  None
                
        """
        gdrn_xy   = np.array([p[:2] for p in gdrn_dict["positions"]])

        # Create directions for the spacial relations
        dir_vec = {
            "front"  : np.array([-offset,  0.0]),
            "behind" : np.array([ offset,  0.0]),
            "left"   : np.array([ 0.0 ,  offset]),
            "right"  : np.array([ 0.0 , -offset]),
        }[relation]

        place_list, score_list = [], []

        # Loop through the reference objects
        for i, p_ref in enumerate(P_ref):
            if p_ref == 0.0:
                # If object cannot be reference, skip
                continue

            ref_xy    = gdrn_xy[i]
            candidate = ref_xy + dir_vec

            # Compute vacancy score for the candidate pose
            occupied_xy = np.delete(gdrn_xy, i, axis=0)
            vacancy     = self.get_vacancy_score(candidate, occupied_xy)

            place_prob  = p_ref * vacancy
            if place_prob == 0.0:
                continue

            # If pose viable, add it to pose list with its prob.
            pose = [candidate[0], candidate[1], table_z]
            place_list.append(pose)
            score_list.append(place_prob)

            self.get_logger().info(
                f"[place-helper] ref {gdrn_dict['names'][i]}  "
                f"-> pose {np.round(pose,3)}  "
                f"(P_ref={p_ref:.2f}, vacancy={vacancy:.2f})"
            )

        # Check if any valid pose found
        if not score_list:
            self.get_logger().warn("[place-lang+gest] no valid place pose found")
            return None
        
        probs = np.asarray(score_list, dtype=float)
        probs /= probs.sum()

        # Create a place pose probs_dict
        return {
            "ref_objs": gdrn_dict['names'],
            "places": place_list,
            "place_probs": probs,
        }

    def align_gestures_to_gdrn(self, gest_dict: dict, gdrn_dict: dict):
        """
        Align the 1-D gesture-probability vector to the order of GDRN objects.

        Any GDRN object that is missing in the gesture list is given
        a gesture affiliation probability of 0.0 

        Returns
            dict {
                "gdrn_names": [...],
                "gdrn_confs": np.ndarray,
                "gest_probs": np.ndarray
            }
        """
        self.get_logger().info("Aligning gesture list to GDRN order")

        gdrn_names = gdrn_dict["names"]
        gdrn_confs = np.asarray(gdrn_dict["confidences"])

        # Build quick lookup for the gesture side
        gest_lookup = {n: p for n, p in zip(gest_dict["names"], gest_dict["probs"])}

        # Assemble gesture probs vector in GDRN order
        gest_probs = np.zeros(len(gdrn_names), dtype=float)
        missing = [] # collect names not gestured at                

        for i, name in enumerate(gdrn_names):
            if name in gest_lookup:
                gest_probs[i] = gest_lookup[name]
            else:
                # prob will stay 0.0
                missing.append(name) 

        if missing:
            self.get_logger().warn(
                f"Gesture list has no entry for: {missing} - "
                "probability set to 0.0 for those objects."
            )

        # Normalise gesture probabilities
        total = gest_probs.sum()
        if total > 0:
            gest_probs /= total
        else:
            self.get_logger().warn("All gesture probabilities are zero.")

        # Notify the user about the result
        self.get_logger().info(
            "Aligned gesture probs to GDRN order:\n"
            f"  names       : {gdrn_names}\n"
            f"  gdrn confs  : {np.round(gdrn_confs, 3)}\n"
            f"  gest probs  : {np.round(gest_probs, 3)}"
        )

        return {
            "gdrn_names": gdrn_names,
            "gdrn_confs": gdrn_confs,
            "gest_probs": gest_probs,
        }

    def publish_results(
        self,
        action: str,
        probs_dict: dict,
        gdrn_objs: list,
        action_conf: float = 1.0,
    ):
        """
        Wraps the reasoning outcome into HRICommand and publish it
        to ROS 2 topic and via ZeroMQ to ROS 1 for motion execution

        Input
            action       : str         'pick' | 'place' | 'move' …
            probs_dict   : dict        {"names": [...], "probs": np.ndarray}
            gdrn_objs    : list[dict]  original GDRN objects (for positions)
            action_conf  : float       confidence of the chosen high-level action
        """

        # Check if reasoning determined something
        if(
        probs_dict is None 
        or  (
            "obj_probs" in probs_dict.keys() and len(probs_dict["obj_probs"]) == 0
            )
        or  (
            "place_probs" in probs_dict.keys() and len(probs_dict["place_probs"]) == 0
            )
        ):
            self.get_logger().warn("publish_results called with empty probs_dict")
            return
        
        obj_names = None
        obj_probs = None
        place_poses = None
        place_probs = None
        
        # Based on the action type, print the reasoning result
        if action == 'pick':
            obj_names = probs_dict["names"]
            obj_probs = probs_dict["obj_probs"]

            target_idx = int(np.argmax(obj_probs))
            target_name = obj_names[target_idx]
            target_pos  = gdrn_objs[target_idx]["position"]

            self.get_logger().info("──────── Reasoning result ────────")
            self.get_logger().info(f" Action  : {action}   (conf={action_conf:.3f})")
            self.get_logger().info(f" Objects : {obj_names}")
            self.get_logger().info(f" Probs   : {np.round(obj_probs, 3)}")
            self.get_logger().info(f" → selected: {target_name}  @  {target_pos}")
            self.get_logger().info("──────────────────────────────────")

        elif action == 'place':
            ref_objects = probs_dict["ref_objs"] if "ref_objs" in probs_dict.keys() else []
            place_poses = probs_dict["places"]
            place_probs = probs_dict["place_probs"]

            place_idx = int(np.argmax(place_probs))
            place_coords = place_poses[place_idx]

            rounded_poses = [[round(c, 3) for c in pose] for pose in place_poses]
            rounded_place_coords = [round(c, 3) for c in place_coords]

            self.get_logger().info("──────── Reasoning result ────────")
            self.get_logger().info(f" Action  : {action}   (conf={action_conf:.3f})")
            self.get_logger().info(f" Ref. objects : {ref_objects}")
            self.get_logger().info(f" Place poses : {rounded_poses}")
            self.get_logger().info(f" Probs : {np.round(place_probs, 3)}")
            self.get_logger().info(f" → selected: {rounded_place_coords}")
            self.get_logger().info("──────────────────────────────────")
        
        elif action == 'move':
            # Pick part
            obj_names   = probs_dict["names"]
            obj_probs   = probs_dict["obj_probs"]
            # Place part
            place_poses = probs_dict["places"]
            place_probs = probs_dict["place_probs"]

            pick_idx  = int(np.argmax(obj_probs))
            place_idx = int(np.argmax(place_probs))

            self.get_logger().info("──────── Reasoning result ────────")
            self.get_logger().info(f" Action  : {action} (conf={action_conf:.3f})")
            self.get_logger().info(f"  ↳ object : {obj_names[pick_idx]}  "
                                f"P={obj_probs[pick_idx]:.3f}")
            self.get_logger().info(f"  ↳ place  : {np.round(place_poses[place_idx],3)}  "
                                f"P={place_probs[place_idx]:.3f}")
            self.get_logger().info("──────────────────────────────────")

        # Assemble complete prob_dict to be sent
        data_dict = {
            "action"       : [action], # action always there
            "action_probs" : [action_conf], # for now always 1.0
            "objects"      : [] if obj_names is None else obj_names, # target object part
            "object_probs" : [] if obj_probs is None else obj_probs.tolist(),
            "place_poses"  : [] if place_poses is None else place_poses, # place pose part
            "place_probs"  : [] if place_probs is None else place_probs.tolist()
        }

        self.get_logger().info(f"sending HRICommand: {data_dict}")

        # Publish the created payload to ROS1 and ROS2
        hri_msg = HRICommand()
        hri_msg.header = Header()
        hri_msg.header.stamp = self.get_clock().now().to_msg()
        hri_msg.header.frame_id = "reasoner"
        hri_msg.data = [json.dumps(data_dict)]

        self.socket.send_string(hri_msg.data[0])
        self.pub.publish(hri_msg)

        # Block further reasoning until motion finished
        with self.waiting_lock:
            self.waiting_for_motion_execution = True

def main():
    """
    The main function when all modalities are working 
    """
    rclpy.init()
    reasoner = Reasoner()
    try:
        rclpy.spin(reasoner)
    except KeyboardInterrupt:
        print("Ending by KeyboardInterrupt")
    finally:
        reasoner.destroy_node()
        rclpy.shutdown()

def tester(first=True, reasoner=None):
    def _test(reasoner: Reasoner):
        # function for testing
        user_input = input(f"Enter Testing setup following the instructions above: ")
        user_input = user_input.strip().split()
        
        input_check = True
        if len(user_input) > 0:
            for item in user_input:
                if item in options or item in sensors:
                    continue
                else:
                    input_check = False
                break
        else:
            input_check = False

        if input_check:
            print(f"Testing setup: '{user_input}' entered")
            reasoner_tester = ReasonerTester(reasoner,user_input,2.5)
            return True
        else:
            print(f"Unknown input: '{user_input}', try again please")
            return False

    options = ['shape', 'color', 'left', 'right']
    sensors = ['deictic', 'gdrn', 'nlp', 'all']

    if first:
        print(f"[TESTING]\nTo test with dummy data from all inputs, enter one of the options: {options}\n")
        print(f"To test with at least one input from sensors input the name(s) of the input, {sensors}")
        print("If you test without nlp from sensor, always add action parameter from options above")
        rclpy.init()
        reasoner = Reasoner()

    try:
        if _test(reasoner):
            rclpy.spin(reasoner)
        else:
            if input("Press 'a' to try again or anything else to exit: ") != 'a':
                return
            else:
                tester(False,reasoner)
    except KeyboardInterrupt:
        print("Ending by User Interrupt")
    finally:
        if first:
            reasoner.destroy_node()
            # rclpy.shutdown()


if __name__ == '__main__':
    tester()
    
