"""Fail if dependencies omit a bound or use a direct URL."""

from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = list(payload["project"]["dependencies"])
    for group in payload["project"].get("optional-dependencies", {}).values():
        dependencies.extend(group)
    for requirement in dependencies:
        if "://" in requirement or requirement.startswith("git+"):
            print(f"URL dependency is not allowed: {requirement}")
            return 1
        if ">=" not in requirement or "<" not in requirement:
            print(f"Dependency needs a lower and upper bound: {requirement}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
