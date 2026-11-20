"""Prepare a local IndicVoices or IndicVoices-R directory after access terms are accepted."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("prepare_kathbath.py")), run_name="__main__")
