import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import time

# Global variables
is_recording = False
recording_thread = None
fs = 44100  # Sample rate
recorded_data = []

# def record_audio():
#     global is_recording, recorded_data
#     recorded_data = []  # Reset recorded data before starting

#     def callback(indata, frames, time, status):
#         if status:
#             print(status)
#         recorded_data.append(indata.copy())

#     with sd.InputStream(samplerate=fs, channels=2, callback=callback):
#         while is_recording:
#             sd.sleep(100)

# def start_recording(duration=5):
#     global is_recording, recording_thread
#     if not is_recording:
#         is_recording = True
#         recording_thread = threading.Thread(target=record_audio)
#         recording_thread.start()
#         print("Recording started...")

# def stop_recording():
#     global is_recording, recording_thread, recorded_data
#     if is_recording:
#         is_recording = False
#         recording_thread.join()  # Wait for recording thread to finish
#         print("Recording stopped...")
#         file_name = save_recording()
#         return file_name
#     return None

# def save_recording():
#     global recorded_data
#     audio_data = np.concatenate(recorded_data, axis=0)  # Combine recorded data
#     file_name = f"recording_{int(time.time())}.wav"
#     wav.write(file_name, fs, audio_data)
#     print(f"Recording saved to {file_name}")
#     return file_name


import subprocess

def record_audio__hacked_when_sounddevice_cannot_find_headset(output_file="recording.wav", duration=5, sound_card=1):
    command = [
        "pasuspender", "--", "arecord", 
        f"-D", f"plughw:{sound_card},0",
        "-f", "cd",
        "-t", "wav",
        "-d", str(duration),
        output_file
    ]
    
    subprocess.run(command, check=True)
