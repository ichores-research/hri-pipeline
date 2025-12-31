
import rclpy
import json
import os
import numpy as np
import re
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from gesture_msgs.msg import HRICommand as HRICommandMSG # Download https://github.com/ichores-research/modality_merging to workspace
from pynput import keyboard

from natural_language_processing.speech_to_text.audio_recorder import record_audio__hacked_when_sounddevice_cannot_find_headset
from natural_language_processing.speech_to_text.whisper_model import TextToSpeechModel
from natural_language_processing.sentence_instruct_transformer.sentence_processor import SentenceProcessor
from natural_language_processing.scene_reader import attach_all_labels

RECORD_TIME = 5
RECORD_NAME = "recording.wav"

HOME_DIR = os.path.expanduser("~")  # define your path to the folder with testing sounds
TEST_SENT_DIR = os.path.join(HOME_DIR, "test_nlp_sentences/recordings") # testing sounds folder name

class NLInputPipePublisher(Node):
    def __init__(self):
        super(NLInputPipePublisher, self).__init__("nlinput_node")
        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(HRICommandMSG, "/modality/nlp", qos_profile)

        self.stt = TextToSpeechModel()
        self.sentence_processor = SentenceProcessor()

        self.GESTURE_DELAY_SLEEP_DURATION = 10 # seconds

    def capitalize_segment(self, text):
        """Capitalizes the first letter of the text segment and lowercases the rest."""
        return text.capitalize() if text else ""

    def separate_sentences(self, sentence):
        """
        Parses a sentence with up to 3 command segments based on connectors
        (but first, first, then, and then, and), determines logical execution order,
        and returns capitalized segments with original indices.

        Args:
            sentence: The input sentence string.

        Returns:
            A list of tuples, where each tuple is (capitalized_segment, original_index).
            The list is sorted according to the logical execution order.
            Handles 'but first' priority and up to two sequential connectors.
            Limits parsing to a maximum of 3 logical segments.
        """
        sentence = sentence.lower()
        ordered_segments = []

        # Searched keyword definition
        keywords_def = [
            (" but first ", "reverse", len(" but first ")),
            (" and then ", "seq", len(" and then ")),
            (" then ", "seq", len(" then ")),
            (" and ", "seq", len(" and ")),
        ]
        first_prefix = "first "
        starts_with_first = sentence.startswith(first_prefix)

        # Get the actual prefix casing ("First ", "first ", etc.)
        original_first_prefix = sentence[:sentence.find(" ") + 1] if starts_with_first else ""
        first_prefix_len = len(original_first_prefix)

        # Find all connector occurrences, make a dict for all of them
        connectors_found = []
        for kw_str, kw_type, kw_len in keywords_def:
            start_search = 0
            while True:
                idx = sentence.find(kw_str, start_search)
                if idx == -1:
                    break
                # Avoid adding 'and' if 'and then' matched at the same spot, preventing duplicate splits
                is_shadowed_by_and_then = False
                if kw_str == " and ":
                    # Check if ' and then ' starts exactly where ' and ' was found
                    if sentence[idx:].startswith(" and then "):
                        is_shadowed_by_and_then = True

                if not is_shadowed_by_and_then:
                    # Store the original casing of the found keyword
                    original_kw = sentence[idx : idx + kw_len]
                    connectors_found.append({
                        "index": idx,
                        "str": original_kw,
                        "type": kw_type,
                        "len": kw_len
                    })
                # Increment search position correctly
                # If shadowed, still advance past ' and ' part to find next potential connector
                # If not shadowed, advance past the found keyword
                start_search = idx + (len(" and ") if is_shadowed_by_and_then else kw_len)


        connectors_found.sort(key=lambda x: x["index"]) # Sort by position

        # Pick up to 2 connectors for splitting
        primary_connectors = []
        reverse_connector_info = None
        seq_connector_count = 0

        # Find 'but first' if it exists, only consider the first one
        for conn in connectors_found:
            if conn["type"] == "reverse":
                reverse_connector_info = conn
                break 
        
        # Add 'but first' as a primary connector if found
        if reverse_connector_info:
            primary_connectors.append(reverse_connector_info)

        # Add up to two sequential connectors from the found list
        for conn in connectors_found:
            # Don't add the 'but first' connector again
            if conn["type"] == "seq" and conn != reverse_connector_info:
                if seq_connector_count < 2:
                    primary_connectors.append(conn)
                    seq_connector_count += 1
                # We need 2 connectors needed overall for 3 segments
                if len(primary_connectors) >= (2 if not reverse_connector_info else 3) :
                    # Search for max 2 seq connectors if " but first " exists; max 3 seq if it does not
                    if len(primary_connectors) >= 3: # Max 3 connectors regardless
                        break

        # Sort conenctors by their position
        primary_connectors.sort(key=lambda x: x["index"]) 

        # Extract elementary commands from sentence
        segments_data = []
        last_pos = first_prefix_len # Start search after "First " if present
        current_original_idx = 0
        bf_segment_original_idx = -1 # Track original index of the segment *after* 'but first'

        # Find the position of 'but first' within the primary connectors list
        reverse_conn_list_idx = -1
        if reverse_connector_info:
            for i, conn in enumerate(primary_connectors):
                if conn == reverse_connector_info:
                    reverse_conn_list_idx = i
                    break

        # Loop through the primary connectors to define segment end points
        for i, conn in enumerate(primary_connectors):
            # Check if we already have 3 segments before processing this connector
            if current_original_idx >= 3:
                break

            segment_text = sentence[last_pos:conn["index"]].strip()
            if segment_text:
                # Determine if this segment is the one that should be moved by 'but first'
                is_the_bf_target_segment = (reverse_conn_list_idx != -1 and i == reverse_conn_list_idx + 1)

                segments_data.append({
                    "text": segment_text,
                    "original_index": current_original_idx
                })
                if is_the_bf_target_segment:
                    # Mark which segment to move
                    bf_segment_original_idx = current_original_idx 

                current_original_idx += 1
            
            # Move past the current connector
            last_pos = conn["index"] + conn["len"] 


        # Add the final segment (after the last processed primary connector)
        if current_original_idx < 3: # Only add if we haven't hit the 3-segment limit
            final_segment_text = sentence[last_pos:].strip()
            if final_segment_text:
                # Check if this final segment is the one targeted by 'but first'
                is_the_bf_target_segment = (reverse_conn_list_idx != -1 and reverse_conn_list_idx == len(primary_connectors) - 1)

                segments_data.append({
                    "text": final_segment_text,
                    "original_index": current_original_idx
                })
                if is_the_bf_target_segment:
                    bf_segment_original_idx = current_original_idx

        # Reorder the segments if 'but first' was present
        bf_segment_data = None
        other_segments_data = []

        if reverse_connector_info and bf_segment_original_idx != -1:
            for seg_data in segments_data:
                # Find the segment marked to be moved
                if seg_data["original_index"] == bf_segment_original_idx:
                    bf_segment_data = seg_data
                else:
                    other_segments_data.append(seg_data)

            # Construct final ordered list: 'but first' segment first, then others
            if bf_segment_data:
                ordered_segments.append(
                    (self.capitalize_segment(bf_segment_data["text"]), bf_segment_data["original_index"])
                )

            # Append others, maintain their original relative order
            other_segments_data.sort(key=lambda x: x["original_index"])
            for seg_data in other_segments_data:
                ordered_segments.append(
                    (self.capitalize_segment(seg_data["text"]), seg_data["original_index"])
                )
        else:
            # No 'but first' or target segment not found, order is sequential
            segments_data.sort(key=lambda x: x["original_index"])
            for seg_data in segments_data:
                ordered_segments.append(
                    (self.capitalize_segment(seg_data["text"]), seg_data["original_index"])
                )


        # Final filtering for empty strings just in case
        return [cmd for cmd in ordered_segments if cmd[0]]

    def forward(self, recording_name: str, override_prompt: str = None):
        """
        Handles the STT processing using Whisper and then the structuralization
        of the elementary commands one by one.

        Args:
            recording_name: The name of language instructions recording saved 
                in a sound file.
            override_prompt: The text rewrite of a language instruction.
                Optional. If present, the STT is skipped and the override_promt
                text is processed instead.

        Publishes a list[dict] variable, where each dict contains structured
        commands in the following keys:
        - action: desired action type (pick, place, move)
        - target_object: target object name
        - target_object2: reference object name
        - action_parameter: action parameter (spatial relations)
        - target_object_color: target object color
        - target_object_color2 reference object color
        - raw_text: Original wording of the command
        - orig_idx: Order of the command in original STT result or override_prompt
        """
        self.get_logger().info("1. Speech to text")
        if override_prompt:
            # simple override for testing
            sentence_text = override_prompt
            self.get_logger().info(f"[NLInputPipePublisher] Override prompt: {sentence_text}")
        else:
            #speech-to-text transformation
            sentence_text = self.stt.forward(recording_name)
            self.get_logger().info(f"Speech-to-text result: {sentence_text}")
        
        self.get_logger().info("2. Sentence processing")

        # Separate sentences to elementary commands
        sentences = self.separate_sentences(sentence_text)
        self.get_logger().info(f"Detected {len(sentences)} senteces: {sentences}")

        predicted_sentences = []

        # Structuralize each elementary command separately
        for exec_idx, (sentence, orig_idx) in enumerate(sentences):
            self.get_logger().info(f"Processing sentence {exec_idx+1}: {sentence}")
            output = self.sentence_processor.predict(sentence)

            for k in output.keys():
                if isinstance(output[k], np.ndarray):
                    output[k] = list(output[k])

            self.get_logger().info(f"Sentence processing result: {output}")

            # Add raw text for possible move gesture pairing disambiguation
            output["raw_text"] = sentence

            # Add the original order of the sentence in the language input for the gesture-language fusion
            output["orig_idx"] = orig_idx
            predicted_sentences.append(output)

        # Publish all elementary commands from one episode at once
        self.get_logger().info(f"Sending sentences: {predicted_sentences}")
        self.pub.publish(HRICommandMSG(data=[str(json.dumps(predicted_sentences))]))
                         
    def on_press(self, key):
        """
        Keyboard event listener
        """
        try:
            # exit the node by pressing esc
            if key == keyboard.Key.esc:
                return False
            
            # Sound recording not functional right now
            # if key == keyboard.Key.space:  # Start recording on space key press
            #     record_audio__hacked_when_sounddevice_cannot_find_headset(RECORD_NAME, duration=RECORD_TIME, sound_card=1)
            #     self.get_logger().info("Processing started")
            #     self.forward(RECORD_NAME)
            #     # start_recording()
            
            # press f2 to process a saved sound file
            if key == keyboard.Key.f2:
                self.get_logger().info("Delay for possible gesture input...")
                time.sleep(15) # time for using gestures
                self.forward(
                    os.path.join(TEST_SENT_DIR, "language_L5f.wav"), # Change the sound file name
                )

            # press f3 to use override prompt
            if key == keyboard.Key.f3:
                self.get_logger().info("Delay for possible gesture input...")
                time.sleep(15) # time for using gestures
                self.forward(
                    os.path.join(TEST_SENT_DIR, "language_L1a.wav"), # arbitrary
                    # "Place it right to mustard" # Change to whatever you want to test
                    # "Pick up a banana"
                    "Pick up this object and place it left to this banana"
                    # "Place it here but first pick up tomato soup"
                    # "Pick up this red object but first move this banana here"
                )
                
        except AttributeError:
            pass

    # def on_release(self, key):
    #     if key == keyboard.Key.space:  # Stop recording on space key release
    #         recording_name = stop_recording()
    #         if recording_name is not None:
    #             print("Processing started")
    #             self.forward(recording_name)
                
    #     if key == keyboard.Key.esc:  # Exit on ESC key release
    #         return False

def main():
    rclpy.init()
    nl_input = NLInputPipePublisher()
    # Listen to keyboard events
    with keyboard.Listener(on_press=nl_input.on_press) as listener: #, on_release=nl_input.on_release) as listener:
        # print(f"\nPress 'space' to start {RECORD_TIME} second recording or \nPress 'alt' for using pre-recorded recording or\nPress 'esc' to exit.")
        print(f"\nPress 'f2' to process the saved sound file\nPress 'f3' to directly use override prompt\nPress 'esc' to exit.")
        listener.join()

if __name__ == "__main__":
    main()

