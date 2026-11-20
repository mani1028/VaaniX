"""Parse every project Python file and reject eval/exec calls."""

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("src", "tests", "scripts")


def main() -> int:
    for folder in ROOTS:
        for path in (ROOT / folder).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                    print(f"{path} uses {node.func.id}")
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
