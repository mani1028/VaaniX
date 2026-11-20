"""Print the open model registry."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models.yaml"


def load_registry() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def main() -> None:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    data = load_registry()
    print(data["product_name"])
    print("scope:", data["scope"])
    for family in ("asr", "tts", "translate"):
        print(f"\n[{family}] default={data[family]['default']}")
        for name, spec in data[family]["backends"].items():
            print(f"  - {name}: {spec.get('id')} [{spec.get('status')}] license={spec.get('license')}")
    print("\ncascade:", data["cascade"]["speech_to_speech"]["steps"])
    print("excluded:", ", ".join(data["excluded"]))


if __name__ == "__main__":
    main()
