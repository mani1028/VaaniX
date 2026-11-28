"""Shim: the real CLI lives in telugu_speech.bench.

    python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te --n 200
    python scripts/bench.py --asr-model telugu-asr-micro --sets fleurs-te --n 200
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_speech.bench import main

if __name__ == "__main__":
    main()
