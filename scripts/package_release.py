from __future__ import annotations
import argparse
import fnmatch
import re
import zipfile
from pathlib import Path
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".coverage", "data", "reports",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES = {".DS_Store", ".coverage"}
EXCLUDED_PATHS = {"eval/results.json", "eval/results.md"}
def read_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not determine package version from pyproject.toml")
    return match.group(1)
def should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return True
    if rel.as_posix() in EXCLUDED_PATHS:
        return True
    if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
        return True
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in ("*.egg-info",)):
        return True
    return False
def build_release(root: Path, output: Path) -> tuple[str, int]:
    version = read_version(root)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == output:
            continue
        if should_exclude(path, root):
            continue
        files.append(path)
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            arcname = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_size = 0
            with path.open("rb") as fh:
                zf.writestr(info, fh.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return version, len(files)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", help="output ZIP path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = read_version(root)
    output = Path(args.output) if args.output else root.parent / f"ResearchPilot-v{version}-ultimate-final.zip"
    built_version, count = build_release(root, output)
    print(f"Built ResearchPilot v{built_version}: {output} ({count} files)")
if __name__ == "__main__":
    main()
