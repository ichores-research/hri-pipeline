import numpy as np
""" TODO! Append all actions and scene objects that are detected!
    This script should read the available action labels and object labels.
    Now the labels are predefined!
"""

A = ['move_up', 'release', 'stop', 'pick_up', 'push', 'unglue', 'pour', 'put', 'stack']
O = ['cup', 'bowl']

def attach_all_labels(output):
    output["action_probs"] = np.zeros(len(A))
    output[output["target_action"]] = 1.0
    output["action"] = A

    output["object_probs"] = np.zeros(len(O))
    output[output["target_object"]] = 1.0
    output["object"] = O

    return output