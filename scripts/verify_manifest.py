"""Create or verify the repository SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def included_files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("data", "input") and path.name != ".gitkeep":
            continue
        result.append(path)
    return sorted(result, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content() -> str:
    return "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in included_files()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Replace MANIFEST.sha256.")
    args = parser.parse_args()
    expected = content()
    if args.write:
        MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Wrote {MANIFEST} ({len(included_files())} files)")
        return
    if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != expected:
        raise SystemExit("MANIFEST.sha256 is missing or stale; run with --write")
    print(f"Manifest verified ({len(included_files())} files)")


if __name__ == "__main__":
    main()
