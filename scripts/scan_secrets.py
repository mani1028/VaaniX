"""Reject obvious secret assignments in the repository source."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]",
    re.IGNORECASE,
)
SKIP = {".venv", ".venv311", ".venv-test", "site-packages", "checkpoints", ".git"}


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP or part.startswith(".venv") for part in path.parts):
            continue
        if path.suffix not in {".py", ".yml", ".yaml", ".md", ".toml", ".example"} and path.name != ".env.example":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PATTERN.search(text):
            failures.append(str(path.relative_to(ROOT)))
    if failures:
        print("Possible secrets:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
