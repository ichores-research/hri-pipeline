#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mic_analyzer_node.py

ROS1 Noetic microphone analyzer:
- Subscribes to audio stream (AudioDataStamped by default, AudioData in legacy mode).
- Subscribes to AudioInfo to learn sample_rate/channels/sample_format.
- Runs WebRTC VAD to segment speech.
- Applies lightweight gates to drop noise / false positives.
- Runs OpenAI Whisper transcription with word timestamps.
- Annotates deictic words/articles/phrases + connector words.
- Optional hot-word activation (one-shot + short continuation window).
- Optional hot-phrase start/stop activation (persistent listen state) with BoolStamped publication.
- Optional publishing of transcripts to ROS topics (raw + segmented).

Backward compatibility:
- `--use-simple-data` / `--no-stamp`: subscribe to audio_common_msgs/AudioData and use receive time.
- Default: audio_common_msgs/AudioDataStamped and use msg.header.stamp for capture-time alignment.

Dependencies:
- webrtcvad
- whisper (openai/whisper)
- scipy
- audio_common_msgs
- input_handling msgs: AnnotWord, SegmentedTranscript, BoolStamped

Notes on "raw" output topic:
- To avoid guessing an unknown custom message, the "raw" output topic publishes the same
  `input_handling/SegmentedTranscript` message but with `words=[]`.
  This still includes start/end stamps and plain text.
"""

import argparse
import os
import re
import time
import wave
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly
import webrtcvad
import whisper

try:
    import rospy
    from audio_common_msgs.msg import AudioData, AudioInfo, AudioDataStamped
    # from input_handling.msg import AnnotWord, SegmentedTranscript, BoolStamped
    ROS_AVAILABLE = True
except Exception:
    rospy = None
    AudioData = None
    AudioInfo = None
    AudioDataStamped = None
    AnnotWord = None
    SegmentedTranscript = None
    BoolStamped = None
    ROS_AVAILABLE = False


def ros_warn(msg: str) -> None:
    if ROS_AVAILABLE:
        rospy.logwarn(msg)
    else:
        print("[WARN] " + msg)


def ros_info(msg: str) -> None:
    if ROS_AVAILABLE:
        rospy.loginfo(msg)
    else:
        print("[INFO] " + msg)


def get_ros_default(ns: str, key: str, fallback: Any) -> Any:
    if not ROS_AVAILABLE:
        ros_warn(f"ROS not available, cannot read param '~{ns}/{key}'")
        return fallback
    full = f"~{ns}/{key}"
    if rospy.has_param(full):
        return rospy.get_param(full)
    ros_warn(f"ROS param '{full}' not set; using fallback {fallback}")
    return fallback


def int16_bytes_to_numpy(data_bytes: bytes) -> np.ndarray:
    return np.frombuffer(data_bytes, dtype=np.int16)


def write_wav(path: str, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    samples_i16 = samples.astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())


def resample_to_target(x_int16: np.ndarray, src_rate: int, tgt_rate: int) -> np.ndarray:
    if src_rate == tgt_rate:
        return x_int16
    gcd = np.gcd(src_rate, tgt_rate)
    up = tgt_rate // gcd
    down = src_rate // gcd
    y = resample_poly(x_int16.astype(np.float32), up, down)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y


def rms_int16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


_PUNCT_RE = re.compile(r"[^\\w\\s\\[\\],]")
_WS_RE = re.compile(r"\\s+")


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_token(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\\w]", "", s)
    return s


def compile_phrase_regex(phrase: str) -> re.Pattern:
    p = normalize_text(phrase)

    def repl(m: re.Match) -> str:
        alts = [a.strip() for a in m.group(1).split(",") if a.strip()]
        if not alts:
            return ""
        return "(" + "|".join(map(re.escape, alts)) + ")"

    p = re.sub(r"\\[([^\\]]+)\\]", repl, p)
    p = r"\\b" + p.replace(" ", r"\\s+") + r"\\b"
    return re.compile(p, flags=re.IGNORECASE)


def phrase_to_token_alternatives(phrase: str) -> List[List[str]]:
    phrase = normalize_text(phrase)
    parts = phrase.split()
    seqs: List[List[str]] = [[]]
    for part in parts:
        m = re.fullmatch(r"\\[([^\\]]+)\\]", part)
        if m:
            alts = [normalize_token(x.strip()) for x in m.group(1).split(",") if x.strip()]
            new_seqs: List[List[str]] = []
            for s in seqs:
                for a in alts:
                    new_seqs.append(s + [a])
            seqs = new_seqs
        else:
            t = normalize_token(part)
            seqs = [s + [t] for s in seqs]
    return [s for s in seqs if s]


def find_subsequence(tokens: List[str], pattern: List[str]) -> Optional[Tuple[int, int]]:
    if not pattern or len(pattern) > len(tokens):
        return None
    for i in range(0, len(tokens) - len(pattern) + 1):
        if tokens[i:i + len(pattern)] == pattern:
            return (i, i + len(pattern))
    return None


def drop_phrase_from_wordlist(words: List["AnnotWord"], phrase_patterns: List[List[str]]) -> List["AnnotWord"]:
    if not words:
        return words
    tokens_norm = [normalize_token(w.word) for w in words]
    for pat in phrase_patterns:
        hit = find_subsequence(tokens_norm, pat)
        if hit:
            a, b = hit
            return words[:a] + words[b:]
    return words


@dataclass
class AnalyzerConfig:
    use_simple_data: bool
    audio_topic: str
    audio_info_topic: str
    input_rate: int
    num_channels: int

    vad_rate: int
    frame_ms: int
    vad_mode: int
    prebuffer_s: float
    hang_in_ms: int
    hang_out_ms: int
    merge_gap_ms: int

    min_transcribe_ms: int
    min_voiced_ratio: float
    snr_gate_mult: float
    noise_ema_alpha: float

    whisper_model: str
    whisper_device: str
    use_gpu_fp16: bool

    save_audio: bool
    recordings_path: str
    raw_text_topic: Optional[str]
    segmented_text_topic: Optional[str]

    deictic_words: List[str]
    deictic_articles: List[str]
    deictic_phrases: List[str]
    connector_words: List[str]

    hot_word_activation: bool
    hot_words: List[str]
    hot_window_s: float
    hotword_prefix_only: bool

    hot_phrases_start: List[str]
    hot_phrases_end: List[str]
    listening_active_default: bool
    hot_phrase_topic: Optional[str]


class AudioAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available. Source your Noetic environment.")

        self.cfg = cfg
        self.lock = RLock()

        if self.cfg.save_audio:
            os.makedirs(self.cfg.recordings_path, exist_ok=True)

        assert self.cfg.frame_ms in (10, 20, 30), "frame_ms must be 10,20,30 for webrtcvad"
        self.frame_samples_in = int(self.cfg.input_rate * (self.cfg.frame_ms / 1000.0))
        self.vad_frame_samples = int(self.cfg.vad_rate * (self.cfg.frame_ms / 1000.0))

        self.prebuffer_max_frames = int(np.ceil(
            self.cfg.prebuffer_s * self.cfg.input_rate / max(1, self.frame_samples_in)))
        self.raw_prebuffer: List[Tuple["rospy.Time", np.ndarray]] = []
        self._prebuffer_max_age = float(self.cfg.prebuffer_s)

        self.residual = np.zeros((0,), dtype=np.int16)
        self.residual_time: Optional["rospy.Time"] = None

        self.vad = webrtcvad.Vad(self.cfg.vad_mode)
        self.state = "IDLE"
        self.segment_frames: List[Tuple["rospy.Time", np.ndarray, bool]] = []
        self.possible_count = 0
        self.silence_count = 0

        self.noise_rms_ema = 300.0

        self.hot_armed_until_ts: float = 0.0
        self.listening_active = bool(self.cfg.listening_active_default)

        if self.cfg.hot_word_activation and not self.cfg.hot_words:
            self.cfg.hot_words = ["diego"]
        self.cfg.hot_words = [normalize_token(w) for w in self.cfg.hot_words if normalize_token(w)]
        self.cfg.connector_words = [normalize_token(w) for w in self.cfg.connector_words if normalize_token(w)]
        self.cfg.deictic_words = [normalize_token(w) for w in self.cfg.deictic_words if normalize_token(w)]
        self.cfg.deictic_articles = [normalize_token(w) for w in self.cfg.deictic_articles if normalize_token(w)]

        self.hot_start_regex = [compile_phrase_regex(p) for p in self.cfg.hot_phrases_start if p.strip()]
        self.hot_end_regex = [compile_phrase_regex(p) for p in self.cfg.hot_phrases_end if p.strip()]
        self.hot_start_token_alts = [phrase_to_token_alternatives(p) for p in self.cfg.hot_phrases_start if p.strip()]
        self.hot_end_token_alts = [phrase_to_token_alternatives(p) for p in self.cfg.hot_phrases_end if p.strip()]

        self.deictic_phrase_tokens: List[List[str]] = []
        for ph in self.cfg.deictic_phrases:
            toks = [normalize_token(x) for x in normalize_text(ph).split() if normalize_token(x)]
            if toks:
                self.deictic_phrase_tokens.append(toks)
        self.deictic_phrase_tokens.sort(key=len, reverse=True)

        device = self._resolve_whisper_device(self.cfg.whisper_device)
        ros_info(f"Loading Whisper model='{self.cfg.whisper_model}' device='{device}'...")
        self.model = whisper.load_model(self.cfg.whisper_model, device=device)

        self.pub_raw = rospy.Publisher(self.cfg.raw_text_topic, SegmentedTranscript, queue_size=10) if self.cfg.raw_text_topic else None
        self.pub_segmented = rospy.Publisher(self.cfg.segmented_text_topic, SegmentedTranscript, queue_size=10) if self.cfg.segmented_text_topic else None
        self.pub_listen_state = rospy.Publisher(self.cfg.hot_phrase_topic, BoolStamped, queue_size=10) if self.cfg.hot_phrase_topic else None

        rospy.Subscriber(self.cfg.audio_info_topic, AudioInfo, self.info_cb, queue_size=1)
        if self.cfg.use_simple_data:
            rospy.Subscriber(self.cfg.audio_topic, AudioData, self.audio_cb_simple, queue_size=50)
        else:
            rospy.Subscriber(self.cfg.audio_topic, AudioDataStamped, self.audio_cb_stamped, queue_size=50)

        if self.pub_listen_state:
            self._publish_listening_state(rospy.Time.now())

    def _resolve_whisper_device(self, choice: str) -> str:
        if choice == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return choice

    def info_cb(self, msg: AudioInfo) -> None:
        with self.lock:
            self.cfg.input_rate = int(msg.sample_rate)
            self.cfg.num_channels = int(msg.channels)
            self.frame_samples_in = int(self.cfg.input_rate * (self.cfg.frame_ms / 1000.0))
            self.vad_frame_samples = int(self.cfg.vad_rate * (self.cfg.frame_ms / 1000.0))
            self.prebuffer_max_frames = int(np.ceil(
                self.cfg.prebuffer_s * self.cfg.input_rate / max(1, self.frame_samples_in)))
        ros_info(f"AudioInfo: {msg.sample_rate} Hz, {msg.channels} ch, fmt={msg.sample_format}")

    def audio_cb_stamped(self, msg: AudioDataStamped) -> None:
        base_time = msg.header.stamp
        data = msg.audio.data
        data_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        arr = int16_bytes_to_numpy(data_bytes)
        self._handle_audio_chunk(arr, base_time)

    def audio_cb_simple(self, msg: AudioData) -> None:
        base_time = rospy.Time.now()
        data = msg.data
        data_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        arr = int16_bytes_to_numpy(data_bytes)
        self._handle_audio_chunk(arr, base_time)

    def _handle_audio_chunk(self, arr_i16: np.ndarray, base_time: "rospy.Time") -> None:
        if self.cfg.num_channels > 1:
            arr_i16 = arr_i16.reshape(-1, self.cfg.num_channels)[:, 0]

        with self.lock:
            self.raw_prebuffer.append((base_time, arr_i16))
            if len(self.raw_prebuffer) > self.prebuffer_max_frames * 10:
                self.raw_prebuffer = self.raw_prebuffer[-self.prebuffer_max_frames:]

        if self.residual.size == 0:
            self.residual_time = base_time

        samples = np.concatenate([self.residual, arr_i16])
        offset = 0
        sr = float(self.cfg.input_rate)

        while offset + self.frame_samples_in <= samples.size:
            block = samples[offset:offset + self.frame_samples_in]
            frame_time = (self.residual_time + rospy.Duration.from_sec(offset / sr)) if self.residual_time else base_time
            offset += self.frame_samples_in
            self.process_frame(block, frame_time)

        self.residual = samples[offset:]
        if self.residual.size == 0:
            self.residual_time = None

    def process_frame(self, in_frame_int16: np.ndarray, frame_time: "rospy.Time") -> None:
        frame_vad = resample_to_target(in_frame_int16, self.cfg.input_rate, self.cfg.vad_rate)
        if frame_vad.size > self.vad_frame_samples:
            frame_vad = frame_vad[:self.vad_frame_samples]
        elif frame_vad.size < self.vad_frame_samples:
            frame_vad = np.concatenate([frame_vad, np.zeros((self.vad_frame_samples - frame_vad.size,), dtype=np.int16)])

        try:
            is_speech = self.vad.is_speech(frame_vad.tobytes(), sample_rate=self.cfg.vad_rate)
        except Exception as e:
            rospy.logwarn_throttle(5, f"webrtcvad error: {e}")
            is_speech = False

        if not is_speech:
            r = rms_int16(frame_vad)
            self.noise_rms_ema = (1.0 - self.cfg.noise_ema_alpha) * self.noise_rms_ema + self.cfg.noise_ema_alpha * r

        hang_in_frames = max(1, int(np.ceil(self.cfg.hang_in_ms / self.cfg.frame_ms)))
        hang_out_frames = max(1, int(np.ceil(self.cfg.hang_out_ms / self.cfg.frame_ms)))

        finalize = False
        frames_to_finalize: List[Tuple["rospy.Time", np.ndarray, bool]] = []

        with self.lock:
            if self.state == "IDLE":
                if is_speech:
                    self.state = "POSSIBLE_SPEECH"
                    self.segment_frames = [(frame_time, in_frame_int16, True)]
                    self.possible_count = 1
                    self.silence_count = 0

            elif self.state == "POSSIBLE_SPEECH":
                self.segment_frames.append((frame_time, in_frame_int16, is_speech))
                if is_speech:
                    self.possible_count += 1
                    if self.possible_count >= hang_in_frames:
                        self.state = "SPEECH"
                else:
                    self.state = "IDLE"
                    self.segment_frames = []
                    self.possible_count = 0

            elif self.state == "SPEECH":
                self.segment_frames.append((frame_time, in_frame_int16, is_speech))
                if is_speech:
                    self.silence_count = 0
                else:
                    self.silence_count += 1
                    if self.silence_count >= hang_out_frames:
                        finalize = True
                        frames_to_finalize = list(self.segment_frames)
                        self.state = "IDLE"
                        self.segment_frames = []
                        self.possible_count = 0
                        self.silence_count = 0
            else:
                self.state = "IDLE"
                self.segment_frames = []
                self.possible_count = 0
                self.silence_count = 0

        if finalize:
            self._finalize_segment_from_frames(frames_to_finalize)

    def _assemble_audio_with_preroll(self, pre_start_time: "rospy.Time", first_frame_time: "rospy.Time",
                                    frames: List[np.ndarray]) -> np.ndarray:
        with self.lock:
            cutoff = rospy.Time.now() - rospy.Duration.from_sec(self._prebuffer_max_age)
            cut_frames: List[Tuple["rospy.Time", np.ndarray]] = []
            pre_chunks: List[np.ndarray] = []
            for (ts, chunk) in self.raw_prebuffer:
                if ts >= cutoff:
                    cut_frames.append((ts, chunk))
                if pre_start_time <= ts < first_frame_time:
                    pre_chunks.append(chunk)
            self.raw_prebuffer = cut_frames

        return np.concatenate(pre_chunks + frames).astype(np.int16) if pre_chunks else np.concatenate(frames).astype(np.int16)

    def _finalize_segment_from_frames(self, segment_frames: List[Tuple["rospy.Time", np.ndarray, bool]]) -> None:
        if not segment_frames:
            return

        times, frames, speech_flags = zip(*segment_frames)

        first_time = times[0]
        pre_roll = rospy.Duration.from_sec(self.cfg.hang_in_ms / 1000.0)
        pre_start_time = first_time - pre_roll
        audio = self._assemble_audio_with_preroll(pre_start_time, first_time, list(frames))
        start_time = pre_start_time
        end_time = times[-1] + rospy.Duration.from_sec(self.cfg.hang_out_ms / 1000.0)

        duration_ms = int(len(audio) / float(self.cfg.input_rate) * 1000.0)
        if duration_ms < self.cfg.min_transcribe_ms:
            return

        voiced_ratio = float(sum(1 for s in speech_flags if s)) / float(len(speech_flags))
        if voiced_ratio < self.cfg.min_voiced_ratio:
            return

        seg_rms = rms_int16(audio)
        if seg_rms < (self.noise_rms_ema * self.cfg.snr_gate_mult):
            return

        if self.cfg.save_audio:
            os.makedirs(self.cfg.recordings_path, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            fname = f"speech_{ts}_{duration_ms}ms.wav"
            fpath = os.path.join(self.cfg.recordings_path, fname)
            try:
                write_wav(fpath, audio, self.cfg.input_rate, channels=1)
                ros_info(f"Saved {fpath}")
            except Exception as e:
                ros_warn(f"Failed to save wav: {e}")

        result = self.transcribe_audio_segment(audio)
        ros_info(f"Transcript: {result.get('text', '<EMPTY>')}")
        seg_msg, words = self.build_segmented_message(result, start_time.to_sec(), end_time.to_sec())

        seg_msg, words = self._apply_hot_phrases(seg_msg, words)
        seg_msg, words, should_publish = self._apply_hotword(seg_msg, words)
        seg_msg.words = words

        ros_info("Transcript [{:.3f}-{:.3f}] dur={}ms publish={} active={} : {}".format(
            seg_msg.start_ts, seg_msg.end_ts, duration_ms, should_publish, self.listening_active,
            seg_msg.text if seg_msg.text else "<EMPTY>"
        ))

        if should_publish and seg_msg.text.strip():
            if self.pub_raw:
                raw = SegmentedTranscript()
                raw.start_ts = seg_msg.start_ts
                raw.end_ts = seg_msg.end_ts
                raw.text = seg_msg.text
                raw.words = []
                self.pub_raw.publish(raw)
            if self.pub_segmented:
                self.pub_segmented.publish(seg_msg)

    def transcribe_audio_segment(self, audio: np.ndarray, dtype=np.int16) -> Dict[str, Any]:
        arr = np.asarray(audio, dtype=dtype)
        if arr.size == 0:
            return {"text": "", "segments": []}
        if (arr.size % self.cfg.num_channels) != 0:
            arr = arr[:arr.size - (arr.size % self.cfg.num_channels)]
        stereo = arr.reshape(-1, self.cfg.num_channels)
        mono_int16 = stereo.mean(axis=1).astype(np.int16)
        TARGET_SR = 16000
        if self.cfg.input_rate != TARGET_SR:
            mono_int16 = resample_to_target(mono_int16, self.cfg.input_rate, TARGET_SR)
        mono_float32 = mono_int16.astype(np.float32) / 32768.0
        return self.model.transcribe(
            mono_float32,
            fp16=bool(self.cfg.use_gpu_fp16),
            language="en",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

    def build_segmented_message(self, result: Dict[str, Any], abs_start_ts: float, abs_end_ts: float
                                ) -> Tuple["SegmentedTranscript", List["AnnotWord"]]:
        text = (result.get("text") or "").strip()
        seg_msg = SegmentedTranscript()
        seg_msg.start_ts = float(abs_start_ts)
        seg_msg.end_ts = float(abs_end_ts)
        seg_msg.text = text
        seg_msg.words = []

        whisper_words: List[Dict[str, Any]] = []
        for seg in (result.get("segments") or []):
            for w in (seg.get("words") or []):
                ww = (w.get("word") or "").strip()
                if not ww:
                    continue
                whisper_words.append({
                    "word": ww,
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", float(w.get("start", 0.0)))),
                })

        words: List[AnnotWord] = []
        for w in whisper_words:
            token_norm = normalize_token(w["word"])
            aw = AnnotWord()
            aw.word = token_norm if token_norm else w["word"].strip()
            aw.start = np.float16(max(0.0, w["start"]))
            aw.end = np.float16(max(0.0, w["end"]))
            aw.is_deictic = False
            aw.is_connector = False
            words.append(aw)

        words = self._annotate(words)
        seg_msg.words = words
        return seg_msg, words

    def _annotate(self, words: List["AnnotWord"]) -> List["AnnotWord"]:
        if not words:
            return words
        conn = set(self.cfg.connector_words)
        deict = set(self.cfg.deictic_words)
        articles = set(self.cfg.deictic_articles)
        tokens = [normalize_token(w.word) for w in words]
        for i, t in enumerate(tokens):
            if t in conn:
                words[i].is_connector = True
            if t in deict or t in articles:
                words[i].is_deictic = True
        for pat in self.deictic_phrase_tokens:
            hit = find_subsequence(tokens, pat)
            if hit:
                a, b = hit
                for i in range(a, b):
                    words[i].is_deictic = True
        return words

    def _publish_listening_state(self, stamp: "rospy.Time") -> None:
        if not self.pub_listen_state:
            return
        msg = BoolStamped()
        msg.header.stamp = stamp
        msg.data = bool(self.listening_active)
        self.pub_listen_state.publish(msg)

    def _apply_hot_phrases(self, seg_msg: "SegmentedTranscript", words: List["AnnotWord"]
                           ) -> Tuple["SegmentedTranscript", List["AnnotWord"]]:
        state_changed = False
        norm = normalize_text(seg_msg.text)

        for rx, token_alts in zip(self.hot_end_regex, self.hot_end_token_alts):
            if rx.search(norm):
                self.listening_active = False
                norm = rx.sub(" ", norm).strip()
                state_changed = True
                for alt in token_alts:
                    words = drop_phrase_from_wordlist(words, [alt])

        for rx, token_alts in zip(self.hot_start_regex, self.hot_start_token_alts):
            if rx.search(norm):
                self.listening_active = True
                norm = rx.sub(" ", norm).strip()
                state_changed = True
                for alt in token_alts:
                    words = drop_phrase_from_wordlist(words, [alt])

        seg_msg.text = norm.strip()
        if state_changed:
            self._publish_listening_state(rospy.Time.from_sec(seg_msg.start_ts))
        return seg_msg, words

    def _strip_hotword_prefix(self, text: str, words: List["AnnotWord"]) -> Tuple[str, List["AnnotWord"], bool]:
        norm = normalize_text(text)
        toks = norm.split()
        if not toks:
            return text, words, False
        first = normalize_token(toks[0])
        if first and first in set(self.cfg.hot_words):
            remainder = " ".join(toks[1:]).strip()
            if words and normalize_token(words[0].word) == first:
                words = words[1:]
            return remainder, words, True
        return text, words, False

    def _apply_hotword(self, seg_msg: "SegmentedTranscript", words: List["AnnotWord"]
                       ) -> Tuple["SegmentedTranscript", List["AnnotWord"], bool]:
        if not seg_msg.text.strip():
            return seg_msg, words, False
        if self.listening_active:
            return seg_msg, words, True
        if not self.cfg.hot_word_activation:
            return seg_msg, words, True

        text2, words2, had_hot = self._strip_hotword_prefix(seg_msg.text, list(words))
        text2 = text2.strip()
        if had_hot:
            self.hot_armed_until_ts = max(self.hot_armed_until_ts, seg_msg.end_ts + float(self.cfg.hot_window_s))
            if not text2:
                seg_msg.text = text2
                return seg_msg, words2, False
            seg_msg.text = text2
            return seg_msg, words2, True

        if seg_msg.start_ts <= self.hot_armed_until_ts:
            return seg_msg, words, True
        return seg_msg, words, False

    def spin(self) -> None:
        rospy.spin()


def list_whisper_models() -> None:
    models = [
        "tiny", "tiny.en", "base", "base.en",
        "small", "small.en", "medium", "medium.en",
        "large-v1", "large-v2", "large-v3", "turbo"
    ]
    print("\\n".join(models))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mic analyzer (VAD + Whisper + annotation + activation).")
    p.add_argument("--audio-topic", default="/audio/raw_dji")
    p.add_argument("--audio-info-topic", default=None)
    p.add_argument("--use-simple-data", "--no-stamp", action="store_true")

    p.add_argument("--whisper-model", default="base.en")
    p.add_argument("--list-whisper-models", action="store_true")
    p.add_argument("--whisper-device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--use-gpu", action="store_true")

    p.add_argument("--input-sample-rate", type=int, default=48000)
    p.add_argument("--num-channels", type=int, default=1)

    p.add_argument("--vad-sample-rate", type=int, default=16000)
    p.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=20)
    p.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2)
    p.add_argument("--prebuffer-seconds", type=float, default=1.0)
    p.add_argument("--hang-in-ms", type=int, default=40)
    p.add_argument("--hang-out-ms", type=int, default=120)
    p.add_argument("--merge-gap-ms", type=int, default=300)

    p.add_argument("--min-transcribe-ms", type=int, default=1000)
    p.add_argument("--min-voiced-ratio", type=float, default=0.35)
    p.add_argument("--snr-gate-mult", type=float, default=2.0)
    p.add_argument("--noise-ema-alpha", type=float, default=0.05)

    p.add_argument("--save-audio", action="store_true")
    p.add_argument("--recordings-path", default="recordings")
    p.add_argument("--raw-text-topic", default=None)
    p.add_argument("--segmented-text-topic", default=None)

    p.add_argument("--deictic-word", action="append", default=[])
    p.add_argument("--deictic-article", action="append", default=[])
    p.add_argument("--deictic-phrase", action="append", default=[])
    p.add_argument("--connector-word", action="append", default=[])

    p.add_argument("--hot-word-activation", "--hwa", action="store_true")
    p.add_argument("--hot-word", "--hot", action="append", default=[])
    p.add_argument("--hot-window-s", type=float, default=2.0)

    p.add_argument("--hot-phrase", "--hp", action="append", default=[])
    p.add_argument("--hot-phrase-end", "--hpe", action="append", default=[])
    p.add_argument("--hot-phrase-topic", "--hpa", default=None)
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
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS not available. Source your Noetic environment.")
    ros_ns = "mic_analyzer"
    if not rospy.core.is_initialized():
        rospy.init_node("mic_analyzer_node", anonymous=False, disable_signals=True)

    parser = build_arg_parser()
    inject_ros_defaults(parser, ros_ns)
    args = parser.parse_args()

    if args.list_whisper_models:
        list_whisper_models()
        return

    audio_info_topic = args.audio_info_topic or (str(args.audio_topic).rstrip("/") + "/info")

    deictic_words = args.deictic_word[:] if args.deictic_word else ["this", "that", "these", "those", "here", "there", "it", "one", "thing"]
    deictic_articles = args.deictic_article[:] if args.deictic_article else ["the"]
    deictic_phrases = args.deictic_phrase[:] if args.deictic_phrase else ["over there"]
    connector_words = args.connector_word[:] if args.connector_word else ["and", "then"]

    hot_word_activation = bool(args.hot_word_activation) or bool(args.hot_word)
    hot_words = args.hot_word[:] if args.hot_word else (["diego"] if hot_word_activation else [])

    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    recordings_path = os.path.join(pkg_dir, str(args.recordings_path))
    if args.save_audio:
        os.makedirs(recordings_path, exist_ok=True)

    cfg = AnalyzerConfig(
        use_simple_data=bool(args.use_simple_data),
        audio_topic=str(args.audio_topic),
        audio_info_topic=str(audio_info_topic),
        input_rate=int(args.input_sample_rate),
        num_channels=int(args.num_channels),

        vad_rate=int(args.vad_sample_rate),
        frame_ms=int(args.frame_ms),
        vad_mode=int(args.vad_mode),
        prebuffer_s=float(args.prebuffer_seconds),
        hang_in_ms=int(args.hang_in_ms),
        hang_out_ms=int(args.hang_out_ms),
        merge_gap_ms=int(args.merge_gap_ms),

        min_transcribe_ms=int(args.min_transcribe_ms),
        min_voiced_ratio=float(args.min_voiced_ratio),
        snr_gate_mult=float(args.snr_gate_mult),
        noise_ema_alpha=float(args.noise_ema_alpha),

        whisper_model=str(args.whisper_model),
        whisper_device=str(args.whisper_device),
        use_gpu_fp16=bool(args.use_gpu),

        save_audio=bool(args.save_audio),
        recordings_path=str(recordings_path),
        raw_text_topic=(str(args.raw_text_topic) if args.raw_text_topic else None),
        segmented_text_topic=(str(args.segmented_text_topic) if args.segmented_text_topic else None),

        deictic_words=deictic_words,
        deictic_articles=deictic_articles,
        deictic_phrases=deictic_phrases,
        connector_words=connector_words,

        hot_word_activation=hot_word_activation,
        hot_words=hot_words,
        hot_window_s=float(args.hot_window_s),
        hotword_prefix_only=True,

        hot_phrases_start=args.hot_phrase[:] if args.hot_phrase else [],
        hot_phrases_end=args.hot_phrase_end[:] if args.hot_phrase_end else [],
        listening_active_default=False,
        hot_phrase_topic=(str(args.hot_phrase_topic) if args.hot_phrase_topic else None),
    )

    node = AudioAnalyzer(cfg)
    node.spin()


if __name__ == "__main__":
    main()