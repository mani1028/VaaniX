"""Prepare a local Shrutilipi directory. This corpus is broadcast news, not conversational Telugu."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("prepare_kathbath.py")), run_name="__main__")
