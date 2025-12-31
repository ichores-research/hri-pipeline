#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adv_mic_publisher_node.py

Advanced microphone publisher for ROS1 Noetic.

Publishes:
  - audio_common_msgs/AudioDataStamped on <audio_topic> (default)
    OR audio_common_msgs/AudioData on <audio_topic> in legacy mode (--use-simple-data / --no-stamp)
  - audio_common_msgs/AudioInfo on <audio_info_topic> (latched)

Audio payload is raw interleaved PCM16 little-endian bytes.

Timestamps (AudioDataStamped):
  - msg.header.stamp is the estimated capture time of the *first sample* in the published chunk.
  - For sounddevice / PyAudio callbacks that provide both ADC time and callback current time,
    stamp is computed as:
        stamp = rospy.Time.now() - (current_time - input_buffer_adc_time)
    This removes most scheduling/publish delay without requiring absolute clock sync.

Argument/parameter precedence:
  1) ROS params under ~adv_mic_publisher/<param> are used as argparse defaults if present.
  2) CLI args override defaults.
  3) Final values are written back to ~adv_mic_publisher/<param> (publisher only).
"""

import argparse
import sys
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple

import numpy as np

# -----------------------
# Optional ROS import
# -----------------------
try:
    import rospy
    from std_msgs.msg import Header
    from audio_common_msgs.msg import AudioData, AudioInfo, AudioDataStamped
    ROS_AVAILABLE = True
except Exception:
    rospy = None
    Header = None
    AudioData = None
    AudioInfo = None
    AudioDataStamped = None
    ROS_AVAILABLE = False


@dataclass
class AudioConfig:
    """@brief Container for audio capture configuration."""
    device: Optional[Union[str, int]]
    samplerate: int
    channels: int
    blocksize: int
    dtype: str
    backend: str  # "sounddevice" or "pyaudio"
    gain: float
    highpass_hz: Optional[float]


# -----------------------
# Utilities
# -----------------------
def ros_warn(msg: str) -> None:
    """@brief Print warning through rospy if available, else stderr."""
    if ROS_AVAILABLE:
        rospy.logwarn(msg)
    else:
        print("[WARN] " + msg, file=sys.stderr)


def get_ros_default(ns: str, key: str, fallback: Any) -> Any:
    """
    @brief Get ROS parameter if available, else fallback.
    @param ns Namespace prefix, e.g. "adv_mic_publisher".
    @param key Parameter key (without ns).
    @param fallback Value if param missing or ROS unavailable.
    """
    if not ROS_AVAILABLE:
        ros_warn("ROS not available, cannot read param '~{}/{}'".format(ns, key))
        return fallback
    full = "~{}/{}".format(ns, key)
    if rospy.has_param(full):
        return rospy.get_param(full)
    ros_warn("ROS param '{}' not set; using fallback {}".format(full, fallback))
    return fallback


def set_ros_param(ns: str, key: str, value: Any) -> None:
    """@brief Set ROS param if ROS available."""
    if ROS_AVAILABLE:
        rospy.set_param("~{}/{}".format(ns, key), value)


def apply_gain(x: np.ndarray, gain: float) -> np.ndarray:
    """@brief Multiply audio by gain."""
    return x if gain == 1.0 else x * gain


def highpass_1st_order(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """
    @brief Simple DC blocker / 1st-order highpass per channel.
    @param x Audio float32 array (N,C).
    """
    if cutoff_hz <= 0:
        return x
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    y = np.empty_like(x)
    y[0, :] = x[0, :]
    for i in range(1, x.shape[0]):
        y[i, :] = alpha * (y[i - 1, :] + x[i, :] - x[i - 1, :])
    return y


def resolve_device_sounddevice(device_spec: Optional[Union[str, int]]) -> Optional[int]:
    """
    @brief Resolve device from int index, exact name, or substring into a concrete sounddevice index.
    @return Device index or None (to let PortAudio pick default).

    Behavior:
      - None -> None (default device)
      - int -> int (validated)
      - str digits -> int
      - str substring -> first matching input-capable device
    """
    import sounddevice as sd

    if device_spec is None:
        return None

    # numeric string
    if isinstance(device_spec, str) and device_spec.isdigit():
        device_spec = int(device_spec)

    if isinstance(device_spec, int):
        try:
            info = sd.query_devices(device_spec)
            if info.get("max_input_channels", 0) <= 0:
                raise ValueError("Device {} has no input channels".format(device_spec))
            return device_spec
        except Exception as e:
            raise ValueError("Invalid device index {}: {}".format(device_spec, e))

    # substring search
    if isinstance(device_spec, str):
        devs = sd.query_devices()
        needle = device_spec.lower()
        for i, d in enumerate(devs):
            name = str(d.get("name", "")).lower()
            if needle in name and d.get("max_input_channels", 0) > 0:
                return i
        raise ValueError("No input device matching substring '{}'".format(device_spec))

    return None


def compute_ros_stamp_from_timeinfo(time_info: Any) -> "rospy.Time":
    """
    @brief Compute best-effort capture timestamp for a chunk using PortAudio/PyAudio time_info.
    @return rospy.Time stamp for the first sample; falls back to rospy.Time.now() if unavailable.
    """
    now = rospy.Time.now()
    if time_info is None:
        return now

    # sounddevice keys: 'inputBufferAdcTime', 'currentTime'
    # pyaudio keys: 'input_buffer_adc_time', 'current_time'
    try:
        if isinstance(time_info, dict):
            if "currentTime" in time_info and "inputBufferAdcTime" in time_info:
                dt = float(time_info["currentTime"] - time_info["inputBufferAdcTime"])
                if dt < 0:
                    dt = 0.0
                return now - rospy.Duration.from_sec(dt)
            if "current_time" in time_info and "input_buffer_adc_time" in time_info:
                dt = float(time_info["current_time"] - time_info["input_buffer_adc_time"])
                if dt < 0:
                    dt = 0.0
                return now - rospy.Duration.from_sec(dt)
    except Exception:
        pass
    return now


# -----------------------
# Audio backends
# -----------------------
class SoundDeviceBackend:
    """
    @brief Capture backend using python-sounddevice (PortAudio).

    The callback pushes (frames, time_info) into a queue.
    frames are float32 (N,C) by default.
    """

    def __init__(self, cfg: AudioConfig, frames_q: queue.Queue):
        import sounddevice as sd
        self.sd = sd
        self.cfg = cfg
        self.frames_q = frames_q
        self.stream = None
        self._stop_evt = threading.Event()

    def list_devices(self) -> None:
        """@brief Print available audio devices."""
        print(self.sd.query_devices())

    def start(self) -> None:
        """@brief Start capture stream."""
        try:
            device_info = self.sd.query_devices(self.cfg.device)
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

        channels = self.cfg.channels or int(device_info.get("max_input_channels", 1))
        samplerate = self.cfg.samplerate or int(device_info.get("default_samplerate", 48000))

        def callback(indata, frames, time_info, status):
            if status:
                print("[sounddevice status] {}".format(status), file=sys.stderr)
            if self._stop_evt.is_set():
                return
            try:
                # Store timing info too (dict-like)
                ti = {"currentTime": time_info.currentTime, "inputBufferAdcTime": time_info.inputBufferAdcTime}
                self.frames_q.put((indata.copy(), ti), block=False)
            except queue.Full:
                pass

        self.stream = self.sd.InputStream(
            device=self.cfg.device,
            channels=channels,
            samplerate=samplerate,
            blocksize=self.cfg.blocksize,
            dtype=self.cfg.dtype,
            callback=callback,
        )
        dev_name = self.sd.query_devices(self.stream.device).get("name", "unknown")
        print("Recording from {} ({} Hz, {} ch)".format(dev_name, self.stream.samplerate, self.stream.channels))
        self.stream.start()

    def stop(self) -> None:
        """@brief Stop capture stream."""
        self._stop_evt.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()


class PyAudioBackend:
    """
    @brief Capture backend using PyAudio.

    Callback pushes (frames_float32, time_info_dict) into a queue.
    """

    def __init__(self, cfg: AudioConfig, frames_q: queue.Queue):
        import pyaudio
        self.pyaudio_mod = pyaudio
        self.pyaudio = pyaudio.PyAudio()
        self.cfg = cfg
        self.frames_q = frames_q
        self.stream = None
        self._stop_evt = threading.Event()

        self.format = pyaudio.paInt16 if cfg.dtype in ("int16", "pcm16") else pyaudio.paFloat32

    def list_devices(self) -> None:
        """@brief Print available PyAudio devices."""
        info = self.pyaudio.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")
        for i in range(numdevices):
            di = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
            print(i, di.get("name"), "inputs:", di.get("maxInputChannels"))

    def start(self) -> None:
        """@brief Start capture stream."""

        def callback(in_data, frame_count, time_info, status):
            if self._stop_evt.is_set():
                return (None, 0)

            if self.format == self.pyaudio_mod.paInt16:
                audio = (np.frombuffer(in_data, dtype=np.int16)
                         .reshape(-1, self.cfg.channels)
                         .astype(np.float32) / 32768.0)
            else:
                audio = (np.frombuffer(in_data, dtype=np.float32)
                         .reshape(-1, self.cfg.channels))

            try:
                self.frames_q.put((audio, dict(time_info) if time_info is not None else None), block=False)
            except queue.Full:
                pass
            return (None, self.pyaudio_mod.paContinue)

        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.cfg.channels,
            rate=self.cfg.samplerate,
            input=True,
            input_device_index=self.cfg.device if isinstance(self.cfg.device, int) else None,
            frames_per_buffer=self.cfg.blocksize,
            stream_callback=callback,
        )
        self.stream.start_stream()

    def stop(self) -> None:
        """@brief Stop stream and terminate PyAudio."""
        self._stop_evt.set()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()


# -----------------------
# ROS publisher
# -----------------------
class RosAudioPublisher:
    """
    @brief Publishes audio stream and metadata.

    Publishes:
      - AudioDataStamped (default) or AudioData (legacy) on audio_topic
      - A latched AudioInfo on info_topic

    In stamped mode, header.stamp must be the capture time of the first sample.
    """

    def __init__(self, audio_topic: str, info_topic: str, queue_size: int, use_stamped: bool):
        self.use_stamped = bool(use_stamped)
        if self.use_stamped:
            self.audio_pub = rospy.Publisher(audio_topic, AudioDataStamped, queue_size=queue_size)
        else:
            self.audio_pub = rospy.Publisher(audio_topic, AudioData, queue_size=queue_size)
        self.info_pub = rospy.Publisher(info_topic, AudioInfo, queue_size=1, latch=True)

        self.audio_topic = audio_topic
        self.info_topic = info_topic

    def publish_info(self, samplerate: int, channels: int) -> None:
        """@brief Publish latched AudioInfo describing the stream."""
        info = AudioInfo()
        info.channels = int(channels)
        info.sample_rate = int(samplerate)
        info.sample_format = "S16LE"
        info.coding_format = "PCM"
        info.bitrate = int(samplerate * channels * 16)
        rospy.loginfo_once("Publishing AudioInfo at {} [{}]".format(self.info_pub.name, self.info_pub.type))
        self.info_pub.publish(info)

    def publish_pcm16le(self, audio_float: np.ndarray, stamp: Optional["rospy.Time"] = None) -> None:
        """@brief Publish one chunk as PCM16LE."""
        audio_i16 = np.clip(audio_float, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        payload = audio_i16.tobytes(order="C")

        rospy.loginfo_once("Publishing at {} [{}]".format(self.audio_pub.name, self.audio_pub.type))

        if not self.use_stamped:
            msg = AudioData()
            # uint8[] accepts bytes; this is faster than list(payload)
            msg.data = payload
            self.audio_pub.publish(msg)
            return

        if stamp is None:
            stamp = rospy.Time.now()

        msg = AudioDataStamped()
        msg.header = Header(stamp=stamp)
        msg.audio = AudioData()
        msg.audio.data = payload
        self.audio_pub.publish(msg)


# -----------------------
# Args / params
# -----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """@brief Create argparse parser; defaults can be injected from ROS params."""
    p = argparse.ArgumentParser(description="Advanced mic publisher for ROS1.")

    p.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    p.add_argument("--backend", choices=["sounddevice", "pyaudio", "auto"], default="auto")
    p.add_argument("--device", "-d", default=None, help="Input device name substring or index.")
    p.add_argument("--rate", type=int, default=48000, help="Sample rate (Hz).")
    p.add_argument("--channels", "--ch", type=int, default=1, help="Number of channels.")
    p.add_argument("--blocksize", type=int, default=1024, help="Frames per callback.")
    p.add_argument("--dtype", default="float32", help="Capture dtype (float32/int16).")
    p.add_argument("--gain", type=float, default=1.0, help="Linear gain.")
    p.add_argument("--highpass", type=float, default=0.0, help="Highpass cutoff Hz (0 disables).")

    p.add_argument("--ros", action="store_true", help="Enable ROS publishing.")
    p.add_argument("--audio-topic", default="/audio/raw_dji", help="Audio topic.")
    p.add_argument("--audio-info-topic", default=None,
                   help="AudioInfo topic. Default: <audio-topic>/info")
    p.add_argument("--ros-queue", type=int, default=10, help="ROS publisher queue.")
    p.add_argument("--ros-rate-limit", type=float, default=0.0,
                   help="Max publish rate Hz (0 = every block).")
    p.add_argument(
        "--use-simple-data", "--no-stamp",
        action="store_true",
        help="Publish legacy AudioData (no capture stamp) instead of AudioDataStamped.",
    )
    return p


def inject_ros_defaults(parser: argparse.ArgumentParser, ros_ns: str) -> None:
    """@brief Override argparse defaults with ROS params if present."""
    ros_to_arg = {
        "use_ros": "ros",
        "audio_topic": "audio_topic",
        "audio_info_topic": "audio_info_topic",
        "input_sample_rate": "rate",
        "num_channels": "channels",
        "blocksize": "blocksize",
        "device": "device",
        "gain": "gain",
        "highpass_hz": "highpass",
        "backend": "backend",
        "dtype": "dtype",
        "ros_queue": "ros_queue",
        "ros_rate_limit": "ros_rate_limit",
    }
    for ros_key, arg_key in ros_to_arg.items():
        default = parser.get_default(arg_key)
        val = get_ros_default(ros_ns, ros_key, default)
        parser.set_defaults(**{arg_key: val})


def select_backend(choice: str) -> str:
    """@brief Choose audio backend."""
    if choice in ("sounddevice", "pyaudio"):
        return choice
    try:
        import sounddevice  # noqa
        return "sounddevice"
    except Exception:
        return "pyaudio"


# -----------------------
# Main
# -----------------------
def main() -> None:
    ros_ns = "adv_mic_publisher"

    # Init ROS early only to read defaults.
    if ROS_AVAILABLE and not rospy.core.is_initialized():
        rospy.init_node("adv_mic_publisher_node", anonymous=True, disable_signals=True)
    else:
        raise RuntimeError("ROS not available. Source ROS first.")

    parser = build_arg_parser()
    inject_ros_defaults(parser, ros_ns)
    args = parser.parse_args()

    backend_name = select_backend(args.backend)

    if args.audio_info_topic is None:
        args.audio_info_topic = args.audio_topic.rstrip("/") + "/info"

    frames_q: "queue.Queue[Tuple[np.ndarray, Optional[Dict[str, Any]]]]" = queue.Queue(maxsize=50)

    # Resolve device before starting stream
    if backend_name != "sounddevice":
        # PyAudio doesn't easily support substring matching without extra code;
        # keep old behavior: int index or None.
        if args.device is not None and str(args.device).isdigit():
            resolved_device: Optional[Union[str, int]] = int(args.device)
        else:
            resolved_device = None
            if isinstance(args.device, str):
                ros_warn("PyAudio backend can't resolve substring device names; using default device.")
    else:
        try:
            resolved_device = resolve_device_sounddevice(args.device)
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    cfg = AudioConfig(
        device=resolved_device,
        samplerate=int(args.rate),
        channels=int(args.channels),
        blocksize=int(args.blocksize),
        dtype=str(args.dtype),
        backend=backend_name,
        gain=float(args.gain),
        highpass_hz=float(args.highpass) if float(args.highpass) > 0 else None,
    )

    backend = SoundDeviceBackend(cfg, frames_q) if backend_name == "sounddevice" else PyAudioBackend(cfg, frames_q)

    if args.list_devices:
        backend.list_devices()
        return

    ros_pub = None
    if args.ros:
        ros_pub = RosAudioPublisher(
            args.audio_topic,
            args.audio_info_topic,
            args.ros_queue,
            use_stamped=(not args.use_simple_data),
        )

    # Start stream
    backend.start()

    actual_rate = int(getattr(backend.stream, "samplerate", cfg.samplerate))
    actual_channels = int(getattr(backend.stream, "channels", cfg.channels))

    # Resolve actual device name/index for params
    actual_device_param = resolved_device
    actual_device_name = None
    if backend_name == "sounddevice":
        import sounddevice as sd
        try:
            di = sd.query_devices(backend.stream.device)
            actual_device_param = int(backend.stream.device)
            actual_device_name = str(di.get("name", ""))
        except Exception:
            pass

    # Publish AudioInfo
    if ros_pub:
        ros_pub.publish_info(actual_rate, actual_channels)

    # Write ROS params (never None)
    set_ros_param(ros_ns, "use_ros", bool(args.ros))
    set_ros_param(ros_ns, "audio_topic", args.audio_topic)
    set_ros_param(ros_ns, "audio_info_topic", args.audio_info_topic)
    set_ros_param(ros_ns, "input_sample_rate", actual_rate)
    set_ros_param(ros_ns, "num_channels", actual_channels)
    set_ros_param(ros_ns, "blocksize", int(args.blocksize))
    set_ros_param(ros_ns, "gain", float(args.gain))
    set_ros_param(ros_ns, "highpass_hz", float(args.highpass))
    set_ros_param(ros_ns, "backend", backend_name)
    set_ros_param(ros_ns, "dtype", args.dtype)
    set_ros_param(ros_ns, "ros_queue", int(args.ros_queue))
    set_ros_param(ros_ns, "ros_rate_limit", float(args.ros_rate_limit))
    set_ros_param(ros_ns, "use_simple_data", bool(args.use_simple_data))

    if actual_device_param is not None:
        set_ros_param(ros_ns, "device", int(actual_device_param))
    else:
        set_ros_param(ros_ns, "device", "")  # explicit default device

    if actual_device_name is not None:
        set_ros_param(ros_ns, "device_name", actual_device_name)

    # Main loop
    last_pub_time = 0.0
    min_pub_period = 0.0 if args.ros_rate_limit <= 0 else (1.0 / float(args.ros_rate_limit))

    try:
        while True:
            try:
                frames, time_info = frames_q.get(timeout=0.1)
            except queue.Empty:
                if rospy.is_shutdown():
                    break
                continue

            frames = apply_gain(frames, cfg.gain)
            if cfg.highpass_hz is not None:
                frames = highpass_1st_order(frames, actual_rate, cfg.highpass_hz)

            if ros_pub:
                wall_now = time.time()
                if min_pub_period == 0.0 or (wall_now - last_pub_time) >= min_pub_period:
                    stamp = None
                    if ros_pub.use_stamped:
                        stamp = compute_ros_stamp_from_timeinfo(time_info)
                    ros_pub.publish_pcm16le(frames, stamp=stamp)
                    last_pub_time = wall_now

            if rospy.is_shutdown():
                break

    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
