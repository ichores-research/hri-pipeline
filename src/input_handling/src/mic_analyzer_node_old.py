#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mic_analyzer_node.py

Microphone analyzer node for ROS1 Noetic: VAD-gated Whisper transcription.

Subscribes:
  - audio_common_msgs/AudioData on <audio_topic>
  - audio_common_msgs/AudioInfo on <audio_info_topic>

Publishes:
  - (currently logs only; hook for transcript topic can be added)

Behavior:
  - Buffers incoming PCM16LE audio.
  - Runs WebRTC VAD on fixed 10/20/30 ms frames (after resampling).
  - Builds speech segments using hang-in/out logic.
  - Applies lightweight gates to reject noise/false positives.
  - Saves WAV and transcribes with Whisper; logs transcript + timestamps + duration.

Argument/parameter precedence:
  1) ROS params under ~mic_analyzer/<param> are used as argparse defaults if present.
  2) CLI args override defaults.

All type hints are Python 3.8 compatible.
"""

import argparse
import os
import wave
import time
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict, Tuple, List
from threading import Lock

import numpy as np
from scipy.signal import resample_poly
import webrtcvad
import whisper

# -----------------------
# Optional ROS import
# -----------------------
try:
    import rospy
    from audio_common_msgs.msg import AudioData, AudioInfo, AudioDataStamped
    ROS_AVAILABLE = True
except Exception:
    rospy = None
    AudioData = None
    AudioInfo = None
    ROS_AVAILABLE = False


# -----------------------
# Utilities
# -----------------------
def ros_warn(msg: str) -> None:
    """@brief Print warning through rospy if available, else stderr."""
    if ROS_AVAILABLE:
        rospy.logwarn(msg)
    else:
        print("[WARN] " + msg)


def get_ros_default(ns: str, key: str, fallback: Any) -> Any:
    """
    @brief Get ROS parameter if available, else fallback.
    @param ns Namespace prefix, e.g. "mic_analyzer".
    @param key Parameter name (without ns).
    """
    if not ROS_AVAILABLE:
        ros_warn("ROS not available, cannot read param '~{}/{}'".format(ns, key))
        return fallback
    full = "~{}/{}".format(ns, key)
    if rospy.has_param(full):
        return rospy.get_param(full)
    ros_warn("ROS param '{}' not set; using fallback {}".format(full, fallback))
    return fallback


def int16_bytes_to_numpy(data_bytes: bytes) -> np.ndarray:
    """@brief Convert raw PCM16LE bytes to numpy int16."""
    return np.frombuffer(data_bytes, dtype=np.int16)


def write_wav(path: str, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """@brief Write int16 samples to WAV file."""
    samples_i16 = samples.astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())


def resample_to_target(x_int16: np.ndarray, src_rate: int, tgt_rate: int) -> np.ndarray:
    """
    @brief Polyphase resample int16 audio to target rate.
    @return int16 signal at tgt_rate.
    """
    if src_rate == tgt_rate:
        return x_int16
    gcd = np.gcd(src_rate, tgt_rate)
    up = tgt_rate // gcd
    down = src_rate // gcd
    y = resample_poly(x_int16.astype(np.float32), up, down)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y


def rms(x: np.ndarray) -> float:
    """@brief Root-mean-square of int16 audio, returned in int16 units."""
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


# -----------------------
# Config container
# -----------------------
@dataclass
class AnalyzerConfig:
    """@brief Runtime configuration for AudioAnalyzer."""
    use_gpu: bool
    input_rate: int
    vad_rate: int
    frame_ms: int
    vad_mode: int
    prebuffer_s: float
    hang_in_ms: int
    hang_out_ms: int
    min_speech_ms: int
    merge_gap_ms: int
    num_channels: int
    audio_topic: str
    audio_info_topic: str
    recordings_path: str
    whisper_model: str
    whisper_device: str  # "cpu" / "cuda" / "auto"

    # New fast gates to reduce empty/noisy segments:
    min_transcribe_ms: int
    min_voiced_ratio: float
    snr_gate_mult: float
    noise_ema_alpha: float


# -----------------------
# Main class
# -----------------------
class AudioAnalyzer:
    """
    @brief Analyzer node: VAD-gated Whisper transcription.

    Receives AudioInfo to set stream expectations, then consumes AudioData.
    Uses fixed VAD frames and a small state machine to segment speech.

    To reduce false positives and empty transcripts, finalized segments are
    filtered by:
      - total duration >= min_transcribe_ms
      - voiced_ratio >= min_voiced_ratio
      - RMS >= noise_floor * snr_gate_mult
    """

    def __init__(self, cfg: AnalyzerConfig):
        if ROS_AVAILABLE and not rospy.core.is_initialized():
            rospy.init_node("mic_analyzer_node", anonymous=False)

        self.cfg = cfg
        self.lock = Lock()

        # Derived sizes for framing.
        assert self.cfg.frame_ms in (10, 20, 30), "frame_ms must be 10,20,30 for webrtcvad"
        self.frame_samples_in = int(self.cfg.input_rate * (self.cfg.frame_ms / 1000.0))
        self.vad_frame_samples = int(self.cfg.vad_rate * (self.cfg.frame_ms / 1000.0))

        # Prebuffer stores recent raw frames for pre-roll.
        self.prebuffer_max_frames = int(np.ceil(
            self.cfg.prebuffer_s * self.cfg.input_rate / self.frame_samples_in))
        self.raw_prebuffer: List[Tuple["rospy.Time", np.ndarray]] = []
        self._prebuffer_max_age = float(self.cfg.prebuffer_s)

        # Residual buffer to join callback chunks into exact VAD frames.
        self.residual = np.zeros((0,), dtype=np.int16)
        self.residual_time = None

        # VAD and state machine.
        self.vad = webrtcvad.Vad(self.cfg.vad_mode)
        self.state = "IDLE"

        # Store frames as (time, frame_int16, is_speech_bool)
        self.segment_frames: List[Tuple["rospy.Time", np.ndarray, bool]] = []
        self.possible_count = 0
        self.silence_count = 0
        self.last_voice_time = None
        self._last_segment = None  # (start, end, audio, path)

        # Hang logic in frames.
        self.hang_in_frames = max(1, int(np.ceil(self.cfg.hang_in_ms / self.cfg.frame_ms)))
        self.hang_out_frames = max(1, int(np.ceil(self.cfg.hang_out_ms / self.cfg.frame_ms)))

        # Running noise floor estimate (RMS) from non-speech frames.
        self.noise_rms_ema = 300.0  # a small nonzero seed (int16 units)

        # Whisper model.
        device = self._resolve_whisper_device(self.cfg.whisper_device)
        self.model = whisper.load_model(self.cfg.whisper_model, device=device)
        rospy.loginfo("Whisper model '{}' loaded, device={}".format(self.cfg.whisper_model, device))

        # ROS subscriptions.
        if ROS_AVAILABLE:
            rospy.Subscriber(self.cfg.audio_info_topic, AudioInfo, self.info_cb, queue_size=1)
            rospy.Subscriber(self.cfg.audio_topic, AudioDataStamped, self.audio_cb, queue_size=50)
            rospy.loginfo("Mic Analyzer node listening to {}, info={}".format(
                self.cfg.audio_topic, self.cfg.audio_info_topic))

        self.info_received = False

    # -----------------------
    # Whisper helpers
    # -----------------------
    def _resolve_whisper_device(self, choice: str) -> str:
        """@brief Resolve whisper device choice."""
        if choice == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return choice

    # -----------------------
    # ROS callbacks
    # -----------------------
    def info_cb(self, msg: AudioInfo) -> None:
        """
        @brief Receive AudioInfo. Updates stream expectations.
        """
        with self.lock:
            self.cfg.input_rate = int(msg.sample_rate)
            self.cfg.num_channels = int(msg.channels)
            if msg.sample_format and msg.sample_format.upper() != "S16LE":
                ros_warn("Unexpected sample_format '{}'; expected S16LE".format(msg.sample_format))
            self.info_received = True

            # Recompute derived sizes after info update.
            self.frame_samples_in = int(self.cfg.input_rate * (self.cfg.frame_ms / 1000.0))
            self.vad_frame_samples = int(self.cfg.vad_rate * (self.cfg.frame_ms / 1000.0))
            self.prebuffer_max_frames = int(np.ceil(
                self.cfg.prebuffer_s * self.cfg.input_rate / self.frame_samples_in))

        rospy.loginfo("AudioInfo received: {} Hz, {} ch, fmt={}".format(
            msg.sample_rate, msg.channels, msg.sample_format))

    def audio_cb(self, msg: AudioDataStamped) -> None:
        """
        @brief Main audio callback.

        - Normalizes msg.data into bytes.
        - Unpacks int16 audio.
        - Prebuffers raw audio.
        - Forms exact-length frames for VAD by accumulating residuals.
        """
        tstamp = rospy.Time.now()

        data = msg.audio.data
        data_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        arr = int16_bytes_to_numpy(data_bytes)

        if self.cfg.num_channels > 1:
            arr = arr.reshape(-1, self.cfg.num_channels)[:, 0]

        with self.lock:
            self.raw_prebuffer.append((tstamp, arr))
            if len(self.raw_prebuffer) > self.prebuffer_max_frames * 10:
                self.raw_prebuffer = self.raw_prebuffer[-self.prebuffer_max_frames:]

        if self.residual.size == 0:
            self.residual_time = tstamp

        samples = np.concatenate([self.residual, arr])
        offset = 0

        while offset + self.frame_samples_in <= samples.size:
            block = samples[offset:offset + self.frame_samples_in]
            offset += self.frame_samples_in

            if self.residual_time is None:
                block_time = tstamp
            else:
                dt_sec = (offset - self.frame_samples_in) / float(self.cfg.input_rate)
                block_time = self.residual_time + rospy.Duration(dt_sec)

            self.process_frame(block, block_time)

        self.residual = samples[offset:]
        if self.residual.size == 0:
            self.residual_time = None

    # -----------------------
    # VAD + state machine
    # -----------------------
    def process_frame(self, in_frame_int16: np.ndarray, ros_time) -> None:
        """
        @brief Run VAD on a fixed-size frame and update speech state.
        """
        frame_vad = resample_to_target(in_frame_int16, self.cfg.input_rate, self.cfg.vad_rate)

        # Enforce exact VAD frame length.
        if frame_vad.size > self.vad_frame_samples:
            frame_vad = frame_vad[:self.vad_frame_samples]
        elif frame_vad.size < self.vad_frame_samples:
            pad = np.zeros((self.vad_frame_samples - frame_vad.size,), dtype=np.int16)
            frame_vad = np.concatenate([frame_vad, pad])

        try:
            is_speech = self.vad.is_speech(frame_vad.tobytes(), sample_rate=self.cfg.vad_rate)
        except Exception as e:
            rospy.logwarn_throttle(5, "webrtcvad error: {}".format(e))
            is_speech = False

        # Update noise floor while idle or on non-speech frames.
        if not is_speech:
            r = rms(frame_vad)
            self.noise_rms_ema = (
                (1.0 - self.cfg.noise_ema_alpha) * self.noise_rms_ema
                + self.cfg.noise_ema_alpha * r
            )

        with self.lock:
            if is_speech:
                self.last_voice_time = ros_time

            if self.state == "IDLE":
                if is_speech:
                    self.state = "POSSIBLE_SPEECH"
                    self.segment_frames = [(ros_time, in_frame_int16, True)]
                    self.possible_count = 1
                    self.silence_count = 0

            elif self.state == "POSSIBLE_SPEECH":
                self.segment_frames.append((ros_time, in_frame_int16, is_speech))
                if is_speech:
                    self.possible_count += 1
                    if self.possible_count >= self.hang_in_frames:
                        self.state = "SPEECH"
                else:
                    # false alarm
                    self.state = "IDLE"
                    self.segment_frames = []
                    self.possible_count = 0

            elif self.state == "SPEECH":
                self.segment_frames.append((ros_time, in_frame_int16, is_speech))
                if is_speech:
                    self.silence_count = 0
                else:
                    self.silence_count += 1
                    if self.silence_count >= self.hang_out_frames:
                        self.finalize_segment()
                        self.state = "IDLE"
                        self.segment_frames = []
                        self.possible_count = 0
                        self.silence_count = 0

            else:
                self.state = "IDLE"

    # -----------------------
    # Segment finalization + filtering
    # -----------------------
    def finalize_segment(self) -> None:
        """
        @brief Build speech segment from buffered frames, gate out noise,
        save WAV, and transcribe.
        """
        if not self.segment_frames:
            return

        times, frames, speech_flags = zip(*self.segment_frames)

        # Compute raw segment duration BEFORE pre-roll.
        raw_samples = int(sum(f.size for f in frames))
        raw_duration_ms = int(raw_samples / float(self.cfg.input_rate) * 1000.0)

        # Voiced ratio inside segment.
        voiced_frames = sum(1 for s in speech_flags if s)
        voiced_ratio = float(voiced_frames) / float(len(speech_flags))

        # Assemble numpy audio (input_rate).
        pre_roll = rospy.Duration(self.cfg.hang_in_ms / 1000.0)
        first_time = times[0]
        pre_start_time = first_time - pre_roll

        pre_frames = []
        pre_times = []
        cutoff = rospy.Time.now() - rospy.Duration(self._prebuffer_max_age)
        cut_frames = []

        for (ts, f) in self.raw_prebuffer:
            if pre_start_time <= ts < first_time:
                pre_frames.append(f)
                pre_times.append(ts)
            if ts >= cutoff:
                cut_frames.append((ts, f))
        self.raw_prebuffer = cut_frames

        if pre_frames:
            audio = np.concatenate(pre_frames + list(frames)).astype(np.int16)
            start_time = pre_times[0]
        else:
            audio = np.concatenate(frames).astype(np.int16)
            start_time = first_time - pre_roll

        end_time = times[-1] + rospy.Duration(self.cfg.hang_out_ms / 1000.0)

        duration_ms = int(len(audio) / float(self.cfg.input_rate) * 1000.0)

        # -------- Gate 1: hard minimum length for transcription --------
        if duration_ms < self.cfg.min_transcribe_ms:
            rospy.loginfo_throttle(
                5,
                "Dropped short segment {}ms (<{}ms)".format(duration_ms, self.cfg.min_transcribe_ms),
            )
            return

        # -------- Gate 2: voiced ratio (fast false-positive killer) -----
        if voiced_ratio < self.cfg.min_voiced_ratio:
            rospy.loginfo_throttle(
                5,
                "Dropped low-voiced segment ratio={:.2f} (<{:.2f})".format(
                    voiced_ratio, self.cfg.min_voiced_ratio
                ),
            )
            return

        # -------- Gate 3: energy vs noise-floor (SNR gate) -------------
        seg_rms = rms(audio)
        if seg_rms < (self.noise_rms_ema * self.cfg.snr_gate_mult):
            rospy.loginfo_throttle(
                5,
                "Dropped low-SNR segment rms={:.1f}, noise={:.1f}, mult={:.2f}".format(
                    seg_rms, self.noise_rms_ema, self.cfg.snr_gate_mult
                ),
            )
            return

        # Merge nearby segments if they are close enough.
        merge_gap_sec = self.cfg.merge_gap_ms / 1000.0
        if self._last_segment is not None:
            last_start, last_end, last_audio, last_path = self._last_segment
            gap = start_time.to_sec() - last_end.to_sec()
            if gap <= merge_gap_sec:
                audio = np.concatenate([last_audio, audio])
                start_time = last_start
                end_time = end_time
                try:
                    if os.path.exists(last_path):
                        os.remove(last_path)
                except Exception:
                    pass

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fname = "speech_{}_{}ms.wav".format(ts, int(len(audio) / float(self.cfg.input_rate) * 1000.0))
        fpath = os.path.join(self.cfg.recordings_path, fname)

        try:
            write_wav(fpath, audio, self.cfg.input_rate, channels=1)
            rospy.loginfo("Saved speech segment {}".format(fpath))
            rospy.loginfo("Duration: {} ms ({:.2f} s), voiced_ratio={:.2f}, rms={:.1f}".format(
                duration_ms, duration_ms / 1000.0, voiced_ratio, seg_rms
            ))
            rospy.loginfo("Time: {:.3f} - {:.3f}".format(start_time.to_sec(), end_time.to_sec()))
            self._last_segment = (start_time, end_time, audio, fpath)
        except Exception as e:
            rospy.logerr("Failed to save wav: {}".format(e))

        self.publish_whisper(audio, start_time, end_time, duration_ms)

    # -----------------------
    # Whisper
    # -----------------------
    def publish_whisper(self, audio: np.ndarray, start_time, end_time, duration_ms: int) -> None:
        """
        @brief Run Whisper and log transcript with timestamps + duration.
        """
        text = self.transcribe_audio_segment(audio)
        rospy.loginfo(
            "Transcript [{:.3f}-{:.3f}] dur={}ms ({:.2f}s): {}".format(
                start_time.to_sec(),
                end_time.to_sec(),
                duration_ms,
                duration_ms / 1000.0,
                text if text else "<EMPTY>"
            )
        )

    def transcribe_audio_segment(self, audio: np.ndarray, dtype=np.int16) -> str:
        """
        @brief Transcribe PCM16LE audio with Whisper.
        Whisper expects 16kHz mono float32 when audio is passed as numpy.
        """
        arr = np.asarray(audio, dtype=dtype)
        if arr.size == 0:
            return ""

        # Ensure whole frames for channel reshape.
        if (arr.size % self.cfg.num_channels) != 0:
            arr = arr[:arr.size - (arr.size % self.cfg.num_channels)]

        # Mix down to mono int16.
        stereo = arr.reshape(-1, self.cfg.num_channels)
        mono_int16 = stereo.mean(axis=1).astype(np.int16)

        # Resample to 16kHz for Whisper (matches CLI).
        TARGET_SR = 16000
        if self.cfg.input_rate != TARGET_SR:
            mono_int16 = resample_to_target(mono_int16, self.cfg.input_rate, TARGET_SR)

        # Normalize to float32 [-1, 1].
        mono_float32 = mono_int16.astype(np.float32) / 32768.0

        # Optional: for short segments, disable previous-text conditioning
        # to avoid weird carryover; keep temperature=0 for stability.
        result = self.model.transcribe(
            mono_float32,
            fp16=self.cfg.use_gpu,
            language="en",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        # rospy.loginfo(f">>> {result.keys()}")
        for seg in result["segments"]:
            for k, v in seg.items():
                rospy.loginfo(f">>> {k}: {v}")
            # rospy.loginfo(f">>> {seg.keys()}")
        # rospy.loginfo(f"Segmented words: {len(result['segments'])}")
        return result.get("text", "").strip()


    def spin(self) -> None:
        """@brief ROS spin."""
        rospy.spin()


# -----------------------
# Args / params
# -----------------------
def list_whisper_models() -> None:
    """@brief Print common Whisper models."""
    models = [
        "tiny", "tiny.en", "base", "base.en",
        "small", "small.en", "medium", "medium.en",
        "large-v1", "large-v2", "large-v3", "turbo"
    ]
    print("\n".join(models))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mic analyzer (VAD + Whisper).")

    # Existing params (ROS-driven defaults)
    p.add_argument("--use-gpu", action="store_true", help="Use fp16 in Whisper.")
    p.add_argument("--input-sample-rate", type=int, default=48000)
    p.add_argument("--vad-sample-rate", type=int, default=16000)
    p.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=20)
    p.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2)
    p.add_argument("--prebuffer-seconds", type=float, default=1.0)
    p.add_argument("--hang-in-ms", type=int, default=40)
    p.add_argument("--hang-out-ms", type=int, default=120)
    p.add_argument("--min-speech-ms", type=int, default=200)  # kept for backwards compatibility
    p.add_argument("--merge-gap-ms", type=int, default=300)
    p.add_argument("--num-channels", type=int, default=1)
    p.add_argument("--audio-topic", default="/audio/raw_dji")
    p.add_argument("--audio-info-topic", default=None)
    p.add_argument("--recordings-path", default="recordings")

    # New non-ROS params
    p.add_argument("--whisper-model", "--model", default="base.en")
    p.add_argument("--list-whisper-models", action="store_true")
    p.add_argument("--whisper-device", choices=["cpu", "cuda", "auto"], default="auto")

    # New fast gating params (no ROS equivalents unless you want them)
    p.add_argument("--min-transcribe-ms", type=int, default=1000,
                   help="Minimum segment length to save/transcribe.")
    p.add_argument("--min-voiced-ratio", type=float, default=0.35,
                   help="Minimum fraction of VAD-positive frames in a segment.")
    p.add_argument("--snr-gate-mult", type=float, default=2.0,
                   help="Segment RMS must exceed noise floor * this multiplier.")
    p.add_argument("--noise-ema-alpha", type=float, default=0.05,
                   help="EMA alpha for noise RMS tracking (0-1).")

    return p


def inject_ros_defaults(parser: argparse.ArgumentParser, ros_ns: str) -> None:
    ros_to_arg = {
        "use_gpu": "use_gpu",
        "input_sample_rate": "input_sample_rate",
        "vad_sample_rate": "vad_sample_rate",
        "frame_ms": "frame_ms",
        "vad_mode": "vad_mode",
        "prebuffer_seconds": "prebuffer_seconds",
        "hang_in_ms": "hang_in_ms",
        "hang_out_ms": "hang_out_ms",
        "min_speech_ms": "min_speech_ms",
        "merge_gap_ms": "merge_gap_ms",
        "num_channels": "num_channels",
        "audio_topic": "audio_topic",
        "audio_info_topic": "audio_info_topic",
        "recordings_path": "recordings_path",
    }
    for ros_key, arg_key in ros_to_arg.items():
        default = parser.get_default(arg_key)
        val = get_ros_default(ros_ns, ros_key, default)
        parser.set_defaults(**{arg_key: val})


def main() -> None:
    ros_ns = "mic_analyzer"

    if ROS_AVAILABLE and not rospy.core.is_initialized():
        rospy.init_node("mic_analyzer_node", anonymous=False, disable_signals=True)

    parser = build_arg_parser()
    inject_ros_defaults(parser, ros_ns)
    args = parser.parse_args()

    if args.list_whisper_models:
        list_whisper_models()
        return

    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    recordings_path = os.path.join(pkg_dir, args.recordings_path)
    os.makedirs(recordings_path, exist_ok=True)

    audio_info_topic = args.audio_info_topic
    if audio_info_topic is None:
        audio_info_topic = args.audio_topic.rstrip("/") + "/info"

    cfg = AnalyzerConfig(
        use_gpu=bool(args.use_gpu),
        input_rate=int(args.input_sample_rate),
        vad_rate=int(args.vad_sample_rate),
        frame_ms=int(args.frame_ms),
        vad_mode=int(args.vad_mode),
        prebuffer_s=float(args.prebuffer_seconds),
        hang_in_ms=int(args.hang_in_ms),
        hang_out_ms=int(args.hang_out_ms),
        min_speech_ms=int(args.min_speech_ms),
        merge_gap_ms=int(args.merge_gap_ms),
        num_channels=int(args.num_channels),
        audio_topic=str(args.audio_topic),
        audio_info_topic=str(audio_info_topic),
        recordings_path=str(recordings_path),
        whisper_model=str(args.whisper_model),
        whisper_device=str(args.whisper_device),

        # new gating params
        min_transcribe_ms=int(args.min_transcribe_ms),
        min_voiced_ratio=float(args.min_voiced_ratio),
        snr_gate_mult=float(args.snr_gate_mult),
        noise_ema_alpha=float(args.noise_ema_alpha),
    )

    node = AudioAnalyzer(cfg)
    node.spin()


if __name__ == "__main__":
    main()
