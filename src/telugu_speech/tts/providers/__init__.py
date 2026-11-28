"""TTS providers — one module per engine, all implementing the same interface.

Importing this package loads NO model and NO engine. Each provider is
imported lazily by ``core.engine`` the first time a voice using that
engine is requested.

Engines:
  * hima-nano  — HimaNanoBackend (the shipped lite engine, Srinivas/Sravani)
  * piper      — PiperBackend (ONNX VITS, CPU-only, recommended upgrade path)
  * onnx-vits  — OnnxVitsBackend (generic VITS ONNX packs)

To add an engine: create ``<name>.py`` with a ``<Name>Backend`` class
subclassing ``BaseTTSBackend``, then add one line to
``core.engine._BACKENDS``. Nothing else changes.
"""

from __future__ import annotations

from .base import BaseTTSBackend

__all__ = ["BaseTTSBackend"]
