"""Package existing baseline files without invoking download-capable build scripts."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from experiments.runner import ROOT, BASELINES


def build(strategy, output):
    source = ROOT / BASELINES[strategy]
    files = [source / "main.py"]
    if strategy == "rag":
        for name in ("docling_models", "embed_model"):
            assets = [p for p in (source / name).rglob("*") if p.is_file()]
            if not assets:
                raise ValueError(f"Missing {name}; prepare assets in Colab first")
            files.extend(assets)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, path.relative_to(source).as_posix())
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=BASELINES)
    parser.add_argument("output")
    args = parser.parse_args()
    print(build(args.strategy, args.output))
