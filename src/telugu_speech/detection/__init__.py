"""Acoustic language identification for routing."""

from telugu_speech.detection.lid import Detection, detect_path, load_detection_config

__all__ = ["Detection", "detect_path", "load_detection_config"]
