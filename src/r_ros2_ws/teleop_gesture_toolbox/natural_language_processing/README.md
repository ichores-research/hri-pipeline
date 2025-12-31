
# Natural language processing

Package requires hri_msgs, right now part of modality_merging package.

Press enter to start voice record, when button is released, the record is processed:
1. Speech to text node (*speech_to_text* folder)
2. Text to command (*sentence_instruct_transformer* folder)

See `nl_node.py` for more details.

## Install

Install packages:
```
conda install -c conda-forge -c pytorch -c robostack-staging pytorch pynput python-sounddevice scipy transformers accelerate ros-humble-desktop
```

## Usage

```
ros2 run natural_language_processing nl_node
```

