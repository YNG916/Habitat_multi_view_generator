#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

from PIL import Image

import _bootstrap  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Select dataset state contact sheets for review")
    parser.add_argument("--root", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sheets = sorted(root.glob("scenes/*/floors/*/regions/*/states/*/contact_sheet.png"))
    selected = random.Random(args.seed).sample(sheets, min(args.num_samples, len(sheets)))
    output = Path(args.output_dir).resolve() if args.output_dir else root / "contact_sheets"
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(selected, start=1):
        Image.open(path).save(output / f"sample_{index:04d}_{path.parent.name}.png")
    print(f"Wrote {len(selected)} review sheets to {output}")


if __name__ == "__main__":
    main()
